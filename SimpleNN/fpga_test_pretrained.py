import numpy as np
import argparse
import pyopencl as cl
import pyopencl.array
from sys import platform
import time
import math
import keras
from keras.datasets import cifar10 as input_data
import pickle
import sys

if platform == "darwin":
    VENDOR_NAME = "apple"
else:
    VENDOR_NAME = "intel"
DEVICE_TYPE = cl.device_type.GPU if VENDOR_NAME == 'apple' or VENDOR_NAME == 'nvidia' else cl.device_type.ACCELERATOR

NN_T = np.float32


class FPGATest:
    hidden_layers = 2
    layers = hidden_layers + 2

    LAYER_FILENAME = "layer_cifar_{0}.p"
    minibatch_size = 10000

    #layer_height = [28 * 28, 512, 512, 10] # MNIST
    layer_height = [32 * 32 * 3, 1024, 512, 512, 10] # CIFAR-10

    learn_rate = 0.005
    regulation_strength = 0.002

    act = {}
    act_cpu = {}
    bias = {}
    weights = {}

    ground_truth = []

    act_buffers = {}
    bias_buffers = {}
    weights_buffers = {}

    def __init__(self, kernel_file):

        if self.layers + 1 != len(self.layer_height):
            print("Bad network config.")
            exit()

        print("Running with {} hidden layers and {} layers total.".format(self.hidden_layers,self.layers))

        print("OpenCL Version v{}".format(".".join([str(i) for i in cl.get_cl_header_version()])))
        print("Finding platform....")
        platform = self.findPlatform(VENDOR_NAME)
        if not platform:
            print("ERROR: Platform not found for name {0}".format(VENDOR_NAME))
            exit(1)

        print("Getting devices...")
        devices = platform.get_devices(device_type=DEVICE_TYPE)
        if len(devices) < 1:
            print("ERROR: No device found for type {0}.".format(DEVICE_TYPE))
            exit(1)

        devices = [devices[1]]

        self.ctx = cl.Context(devices=devices)

        if DEVICE_TYPE == cl.device_type.ACCELERATOR:
            print("Reading binary...")
            binary = kernel_file.read()

            binaries = [binary] * len(devices)

            print("Building...")
            program = cl.Program(self.ctx, devices, binaries)
        else:
            print("Reading program...")
            binary = kernel_file.read()

            program = cl.Program(self.ctx, binary.decode('utf-8')).build()

        self.kForward = program.forward
        self.kForwardSoftMax = program.forward_softmax
        # self.kBackwardFirstDelta = program.backward_first_delta
        # self.kBackward = program.backward

        self.kForward.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32])
        self.kForwardSoftMax.set_scalar_arg_dtypes([None, np.int32, np.int32])
        # self.kBackwardFirstDelta.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32])
        # self.kBackward.set_scalar_arg_dtypes(
        #    [None, None, None, None, NN_T, NN_T, np.int32, np.int32, np.int32])

        self.queue = cl.CommandQueue(self.ctx)

        print("Loading data...")
        _, (self.x_test, self.y_test) = input_data.load_data()

        self.y_test = self.y_test.reshape((10000,))
        self.x_test = self.x_test.reshape(10000, self.layer_height[0])
        self.x_test = self.x_test.astype('float32')
        self.x_test /= 255

        self.correct_pred_fpga = 0
        self.wrong_pred_fpga = 0
        self.correct_pred_cpu = 0
        self.wrong_pred_cpu = 0

    def aligned(self, a, alignment=16):
        if (a.ctypes.data % alignment) == 0:
            return a

        extra = alignment / a.itemsize
        extra = int(round(extra))
        buf = np.empty(a.size + extra, dtype=a.dtype)
        ofs = (-buf.ctypes.data % alignment) / a.itemsize
        ofs = int(round(ofs))
        aa = buf[ofs:ofs + a.size].reshape(a.shape)
        np.copyto(aa, a)
        assert (aa.ctypes.data % alignment) == 0
        return aa

    def print_c_array(self, name, array):
        sys.stdout.write("nn_t {}[] = {{".format(name))
        for x in np.nditer(array):
            sys.stdout.write(" {0:.16e},".format(x))
        print(" };")

    def findPlatform(self, strname):
        platforms = cl.get_platforms()
        for platform in platforms:
            if VENDOR_NAME in platform.get_info(cl.platform_info.NAME).lower():
                return platform
        return None

    def cpu_forward(self, act, weights, bias):
        return act.dot(weights) + bias

    def cpu_relu(self, act):
        act[act < 0] = NN_T(0)
        return act

    def cpu_softmax(self, act):
        # print(act)
        # print(np.max(act, axis=1, keepdims=True))
        e_act = np.exp(act - np.max(act, axis=1, keepdims=True))
        return e_act / np.sum(e_act, axis=1, keepdims=True)

    def cpu_first_delta(self, probs, ground_truth):
        return probs - ground_truth

    def cpu_backward(self, act_prev, weights, bias, delta_next, learn_rate, regulation_strength, minibatch_size):
        dW = act_prev.T.dot(delta_next) / minibatch_size  # or rows_in or rows_out
        db = np.sum(delta_next, axis=0, keepdims=True)
        delta = delta_next.dot(weights.T) * (NN_T(1) - np.power(act_prev, 2))

        dW += regulation_strength * weights

        weights += -learn_rate * dW
        bias += -learn_rate * db
        return weights, bias, delta

    def fpga_function(self, benchmark=False):
        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            is_last_layer = layer+1 == self.layers

            print("Running layer {} ({}->{}) kernel, with relu: {}".format(layer,rows_in,rows_out, 1 if not is_last_layer else 0))
            self.kForward.set_args(self.act_buffers[layer], self.weights_buffers[layer], self.bias_buffers[layer], self.act_buffers[layer+1],
                                   self.minibatch_size, rows_in, rows_out, 1 if not is_last_layer else 0)
            cl.enqueue_task(self.queue, self.kForward).wait()

            if is_last_layer:
                print("Running layer {} softmax kernel".format(layer))
                self.kForwardSoftMax.set_args(self.act_buffers[layer+1],
                                              self.minibatch_size, rows_out)
                cl.enqueue_task(self.queue, self.kForwardSoftMax).wait()

    def cpu_function(self, benchmark=False):
        for layer in range(0, self.layers):
            self.act_cpu[layer+1] = self.cpu_forward(self.act_cpu[layer], self.weights[layer], self.bias[layer])
            if layer+1 != self.layers:
                self.act_cpu[layer + 1] = self.cpu_relu(self.act_cpu[layer+1])
            else:
                self.act_cpu[layer + 1] = self.cpu_softmax(self.act_cpu[layer + 1])

    def getops(self):
        # Bias additions and broadcast plus matrix mult.
        ops = 0
        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            ops += rows_out * self.minibatch_size + (2 * rows_in - 1) * self.minibatch_size * rows_out

        # Final softmax
        ops += self.layer_height[self.layers] * (4 + 100) + 1
        print("Total operations: {0} billion per batch.".format(ops / 1e9))
        return ops

    def create_buffers(self):
        print("Generating/loading data...")

        ram_act = 0
        ram_weights = 0
        ram_bias = 0

        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            size_act = (self.minibatch_size, rows_in)
            size_weights = (rows_in, rows_out)
            size_bias = (1, rows_out)

            # Layer input
            self.act[layer] = self.aligned(np.zeros(size_act, NN_T), alignment=64)
            self.act_cpu[layer] = self.aligned(np.zeros(size_act, NN_T), alignment=64)

            # Load layer file
            layer_file = pickle.load(open(self.LAYER_FILENAME.format(layer), 'rb'))
            # Layer weights
            self.weights[layer] = self.aligned(layer_file['weights'].reshape(size_weights), alignment=64)
            self.bias[layer] = self.aligned(layer_file['bias'].reshape(size_bias), alignment=64)

            # Memory
            ram_act += np.size(self.act[layer]) * 4 / 1e6
            ram_weights += np.size(self.weights[layer]) * 4 / 1e6
            ram_bias += np.size(self.bias[layer]) * 4 / 1e6

        final_output_size = (self.minibatch_size, self.layer_height[self.layers])
        # Final output
        self.act[self.layers] = self.aligned(np.zeros(final_output_size, NN_T), alignment=64)
        self.act_cpu[self.layers] = self.aligned(np.zeros(final_output_size, NN_T), alignment=64)

        # Memory
        ram_act += np.size(self.act[self.layers]) * 4 / 1e6

        print("Activation total memory: {} MB".format(ram_act))
        print("Weights total memory: {} MB".format(ram_weights))
        print("Bias total memory: {} MB".format(ram_bias))

    def set_input(self, batch):
        self.act[0] = self.aligned(self.x_test[batch * self.minibatch_size:(batch + 1) * self.minibatch_size, :],
                                   alignment=64)
        self.act_cpu[0] = self.aligned(self.x_test[batch * self.minibatch_size:(batch + 1) * self.minibatch_size, :],
                                   alignment=64)


        self.ground_truth = self.y_test[batch * self.minibatch_size:(batch + 1) * self.minibatch_size]

        cl.enqueue_copy(self.queue, self.act_buffers[0], self.act[0]).wait()

    def send_buffers_to_device(self):
        mf = cl.mem_flags

        print("Transferring all buffers to device...")

        for layer in range(0, self.layers):
            self.act_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.act[layer])
            self.bias_buffers[layer] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bias[layer])
            self.weights_buffers[layer] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.aligned(self.weights[layer].T.copy(), alignment=64))
        # Final output
        self.act_buffers[self.layers] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.act[self.layers])

    def retrieve_buffers_from_device(self, benchmark=False):
        if not benchmark:
            print("Transferring final result from device...")

        cl.enqueue_copy(self.queue, self.act[self.layers], self.act_buffers[self.layers]).wait()

    def finish_device_queue(self):
        print("Finishing queue..")
        self.queue.finish()

    def count_accuracy(self):
        prediction_fpga = np.argmax(self.act[self.layers], axis=1)
        prediction_cpu = np.argmax(self.act_cpu[self.layers], axis=1)

        print(prediction_cpu)
        print(prediction_fpga)
        print(self.ground_truth)

        self.correct_pred_fpga += np.sum(prediction_fpga == self.ground_truth)
        self.wrong_pred_fpga += np.sum(prediction_fpga != self.ground_truth)

        self.correct_pred_cpu += np.sum(prediction_cpu == self.ground_truth)
        self.wrong_pred_cpu += np.sum(prediction_cpu != self.ground_truth)

    def verify_results(self):
        print("Result FPGA:")
        print("Shape FPGA Result: {}".format(self.act[self.layers].shape))
        print(self.act[self.layers])

        print("Result CPU:")

        # result[result < 0] = 0
        print("Shape CPU Result: {}".format(self.act_cpu[self.layers].shape))
        print(self.act_cpu[self.layers])
        do_benchmark = False
        # finally use np.allclose
        if np.array_equal(self.act_cpu[self.layers], self.act[self.layers]):
            print("EXACT FPGA CPU COMPARISON PASSED!!")
        else:
            print("EXACT CPU COMPARISON DID NOT PASS.")
            print("Max Error: {0}".format(np.max(np.abs(self.act_cpu[self.layers] - self.act[self.layers]))))
            print("Min Error: {0}".format(np.min(np.abs(self.act_cpu[self.layers] - self.act[self.layers]))))

        if np.allclose(self.act_cpu[self.layers], self.act[self.layers]):
            print("FPGA CPU COMPARISON PASSED!!")
            return True
        else:
            print("FPGA CPU COMPARISON DID NOT PASS.")
            print("Max Error: {0}".format(np.max(np.abs(self.act_cpu[self.layers] - self.act[self.layers]))))
            print("Min Error: {0}".format(np.min(np.abs(self.act_cpu[self.layers] - self.act[self.layers]))))
            return False

    def print_accuracy(self):
        print("CPU accuracy: {:.2%}".format(self.correct_pred_cpu / (self.correct_pred_cpu + self.wrong_pred_cpu)))
        print("FPGA accuracy: {:.2%}".format(self.correct_pred_fpga / (self.correct_pred_fpga + self.wrong_pred_fpga)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPGA bitstream loading from python")
    parser.add_argument("bitstream", type=argparse.FileType('rb'))

    args = parser.parse_args()
    fpga_class = FPGATest(args.bitstream)
    fpga_class.create_buffers()
    fpga_class.send_buffers_to_device()
    batches = int(fpga_class.x_test.shape[0] / fpga_class.minibatch_size)
    print("Running {} batches...".format(batches))
    fpga_time = 0.0
    cpu_time = 0.0
    for i in range(0, batches):
        fpga_class.set_input(i)
        start_time = time.time()

        fpga_class.fpga_function(True)

        fpga_time += time.time() - start_time
        fpga_class.retrieve_buffers_from_device()
        start_time = time.time()

        fpga_class.cpu_function(True)

        cpu_time += time.time() - start_time
        fpga_class.count_accuracy()
        if i % 10 == 0:
            print("Batch {}/{} done.\n".format(i + 1, batches))
            fpga_class.print_accuracy()

    print("Batch {0}/{0} done.\n".format(batches))
    fpga_class.print_accuracy()
    fpga_class.finish_device_queue()

    ops = fpga_class.getops()
    print("FPGA Time: {0} usec. {1:.3f} GFLOPS".format(fpga_time / batches * 1e6,
                                                       ops / 1e9 / (fpga_time / batches)))
    print("CPU Time: {0} usec. {1:.3f} GFLOPS".format(cpu_time / batches * 1e6,
                                                      ops / 1e9 / (cpu_time / batches)))

    # if fpga_class.verify_results() and False:
    #    fpga_class.benchmark()
