import numpy as np
import argparse
import pyopencl as cl
import pyopencl.array
from sys import platform
import time
import math
import keras
from keras.datasets import mnist
import pickle
import sys

import itertools

if platform == "darwin":
    VENDOR_NAME = "apple"
else:
    VENDOR_NAME = "intel"
DEVICE_TYPE = cl.device_type.GPU if VENDOR_NAME == 'apple' or VENDOR_NAME == 'nvidia' else cl.device_type.ACCELERATOR


NN_T = np.float32

class FPGATest:
    minibatch_size = 128
    rows_in_1 = 28*28
    rows_out_1 = 128

    rows_in_2 = rows_out_1
    rows_out_2 = 10

    fpga_iter = 20
    cpu_iter = 20

    data_max = 1
    data_min = -data_max

    learn_rate = 0.005
    regulation_strength = 0.002

    act_1 = []
    act_2 = []
    bias_1 = []
    bias_2 = []
    weights_1 = []
    weights_2 = []
    act_out_2 = []
    act_out_2_cpu = []
    ground_truth = []

    act_1Array = None
    act_2Array = None
    bias_1Array = None
    bias_2Array = None
    weights_1Array = None
    weights_2Array = None
    act_out_2Array = None

    def __init__(self, kernel_file):
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
        self.kBackWardFirstDelta = program.backward_first_delta
        self.kBackward = program.backward

        self.kForward.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32])
        self.kForwardSoftMax.set_scalar_arg_dtypes([None, np.int32, np.int32])
        self.kBackWardFirstDelta.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32])
        self.kBackward.set_scalar_arg_dtypes(
            [None, None, None, None, NN_T, NN_T, np.int32, np.int32, np.int32])

        self.queue = cl.CommandQueue(self.ctx)

        print("Loading MNIST...")
        _, (self.x_test, self.y_test) = mnist.load_data()

        self.x_test = self.x_test.reshape(10000, self.rows_in_1)
        self.x_test = self.x_test.astype('float32')
        self.x_test /= 255
        #print(self.x_test[0])

        #self.y_test = keras.utils.to_categorical(self.y_test, self.rows_out_2)
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

    def print_c_array(self,name,array):
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
        #print(act)
        #print(np.max(act, axis=1, keepdims=True))
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
        if not benchmark:
            print("Kernel Forward, setting arguments (layer 1)...")

        self.kForward.set_args(self.act_1Array, self.weights_1Array, self.bias_1Array, self.act_2Array,
                               self.minibatch_size, self.rows_in_1, self.rows_out_1, 1)

        if not benchmark:
            print("Kernel Forward, dispatching task (layer 1)...")

        event = cl.enqueue_task(self.queue, self.kForward)
        event.wait()
        cl.enqueue_read_buffer(self.queue, self.act_2Array, self.act_2).wait()
        #if not benchmark:
            #print(self.act_2)

        if not benchmark:
            print("Kernel Forward, setting arguments (layer 2)...")

        self.kForward.set_args(self.act_2Array, self.weights_2Array, self.bias_2Array,
                               self.act_out_2Array,
                               self.minibatch_size, self.rows_in_2, self.rows_out_2, 0)

        if not benchmark:
            print("Kernel Forward, dispatching task (layer 2)...")

        event = cl.enqueue_task(self.queue, self.kForward)
        event.wait()
        #cl.enqueue_read_buffer(self.queue, self.act_out_2Array, self.act_out_2).wait()
        #print("FPGA buffer before softmax.")
        #print(self.act_out_2)
        if not benchmark:
            print("Kernel ForwardSoftMax, setting arguments (layer 2)...")

        self.kForwardSoftMax.set_args(self.act_out_2Array,
                               self.minibatch_size, self.rows_out_2)

        if not benchmark:
            print("Kernel ForwardSoftMax, dispatching task (layer 2)...")

        event = cl.enqueue_task(self.queue, self.kForwardSoftMax)
        event.wait()


    def cpu_function(self, benchmark=False):

        #self.print_c_array("act",self.act_1)
        #self.print_c_array("bias", self.bias_1)
        #self.print_c_array("weights", self.aligned(self.weights_1.T.copy(), alignment=64))
        #self.print_c_array("act_out", self.act_2)
        #self.print_c_array("act_out_gpu", self.act_2)
        result_1 = self.cpu_forward(self.act_1, self.weights_1, self.bias_1)

        #print(result_1)
        result_1 = self.cpu_relu(result_1)
        #self.print_c_array("act_out_cpu", result_1)
        #if not benchmark:
        #    print(result_1)
        result_2 = self.cpu_forward(result_1, self.weights_2, self.bias_2)
        #print("CPU before softmax")
        #print(result_2)
        self.act_out_2_cpu = self.cpu_softmax(result_2)
        #print(self.act_out_2_cpu)

    def getops(self):
        # Bias additions and broadcast plus matrix mult.
        ops = self.rows_out_1 * self.minibatch_size + (2 * self.rows_in_1 - 1) * self.minibatch_size * self.rows_out_1
        ops += self.rows_out_2 * self.minibatch_size + (2 * self.rows_in_2 - 1) * self.minibatch_size * self.rows_out_2
        print("Total operations: {0} billion.".format(ops / 1e9))
        return ops

    def create_buffers(self):

        size_act_1 = (self.minibatch_size, self.rows_in_1)
        size_weights_1 = (self.rows_in_1, self.rows_out_1)
        size_bias_1 = (1, self.rows_out_1)
        # size_act_out_1 = (minibatch_size, rows_out_1)

        size_act_2 = (self.minibatch_size, self.rows_in_2)
        size_weights_2 = (self.rows_in_2, self.rows_out_2)
        size_bias_2 = (1, self.rows_out_2)
        size_act_out_2 = (self.minibatch_size, self.rows_out_2)

        print("Generating/loading data...")
        np.random.seed(1)
        self.act_1 = self.aligned(np.zeros(size_act_1, NN_T), alignment=64)
        self.act_2 = self.aligned(np.zeros(size_act_2, NN_T), alignment=64)
        #self.bias_1 = self.aligned(np.random.uniform(self.data_min, self.data_max, size=size_bias_1).astype(NN_T), alignment=64)
        #self.bias_2 = self.aligned(np.random.uniform(self.data_min, self.data_max, size=size_bias_2).astype(NN_T), alignment=64)
        #self.weights_1 = self.aligned(np.random.uniform(self.data_min, self.data_max, size=size_weights_1).astype(NN_T), alignment=64)
        #self.weights_2 = self.aligned(np.random.uniform(self.data_min, self.data_max, size=size_weights_2).astype(NN_T), alignment=64)

        tmp_l0 = pickle.load(open('layer_h128_0.p', 'rb'))
        self.bias_1 = tmp_l0['bias'].reshape(size_bias_1)
        self.weights_1 = tmp_l0['weights'].reshape(size_weights_1)
        tmp_l1 = pickle.load(open('layer_h128_1.p', 'rb'))
        self.bias_2 = tmp_l1['bias'].reshape(size_bias_2)
        self.weights_2 = tmp_l1['weights'].reshape(size_weights_2)

        self.act_out_2 = self.aligned(np.zeros(size_act_out_2, NN_T), alignment=64)
        self.act_out_2_cpu = np.zeros(size_act_out_2, NN_T)

        print("Shape Act_1: {}".format(self.act_1.shape))
        print("Memfoot print Act_1: {} MB".format(self.act_1.shape[0] * self.act_1.shape[1] * 4 / 1e6))
        #print(self.act_1)
        print("Shape Bias_1: {}".format(self.bias_1.shape))
        print("Memfoot print Bias_1: {} MB".format(self.bias_1.shape[0] * self.bias_1.shape[1] * 4 / 1e6))
        #print(self.bias_1)
        print("Shape Weights_1.T: {}".format(self.weights_1.T.shape))
        print("Memfoot print Weights_1.T: {} MB".format(self.weights_1.shape[0] * self.weights_1.shape[1] * 4 / 1e6))
        #print(self.weights_1.T)
        print("Shape Act_2: {}".format(self.act_2.shape))
        print("Memfoot print Act_2: {} MB".format(self.act_2.shape[0] * self.act_2.shape[1] * 4 / 1e6))
        #print(self.act_2)
        print("Shape Bias_2: {}".format(self.bias_2.shape))
        print("Memfoot print Bias_2: {} MB".format(self.bias_2.shape[0] * self.bias_2.shape[1] * 4 / 1e6))
        #print(self.bias_2)
        print("Shape Weights_2.T: {}".format(self.weights_2.T.shape))
        print("Memfoot print Weights_2.T: {} MB".format(self.weights_2.shape[0] * self.weights_2.shape[1] * 4 / 1e6))
        #print(self.weights_2.T)
        print("Shape ActOut_2: {}".format(self.act_out_2.shape))
        print("Memfoot print ActOut_2: {} MB".format(self.act_out_2.shape[0] * self.act_out_2.shape[1] * 4 / 1e6))

    def set_input(self, batch):
        #print(self.x_test[batch*self.minibatch_size:(batch+1)*self.minibatch_size, :, :].reshape((self.minibatch_size, self.rows_in_1)).shape)
        self.act_1 = self.x_test[batch*self.minibatch_size:(batch+1)*self.minibatch_size, :]
        #print(self.y_test[batch*self.minibatch_size:(batch+1)*self.minibatch_size].shape)
        self.ground_truth = self.y_test[batch*self.minibatch_size:(batch+1)*self.minibatch_size]
        mf = cl.mem_flags
        cl.enqueue_copy(self.queue, self.act_1Array, self.act_1).wait()

    def send_buffers_to_device(self):
        mf = cl.mem_flags
        print("Transferring act to device...")
        self.act_1Array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.act_1)#cl.array.to_device(self.queue, self.act_1)
        self.act_2Array = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.act_2)
        print("Transferring bias to device...")
        self.bias_1Array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bias_1)
        self.bias_2Array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.bias_2)
        #print("Transferring weights to device...")
        #self.weights_1Array = cl.array.to_device(self.queue, self.weights_1)
        #self.weights_2Array = cl.array.to_device(self.queue, self.weights_2)
        print("Transferring weights transposed to device...")
        self.weights_1Array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.aligned(self.weights_1.T.copy(), alignment=64))
        self.weights_2Array = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.aligned(self.weights_2.T.copy(), alignment=64))
        print("Transferring act_out to device...")
        self.act_out_2Array = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.act_out_2)

    def retrieve_buffers_from_device(self):
        print("Transferring act_out from device...")

        cl.enqueue_copy(self.queue, self.act_out_2, self.act_out_2Array).wait()

    def finish_device_queue(self):
        print("Finishing queue..")
        self.queue.finish()

    def count_accuracy(self):
        self.correct_pred_fpga = 0
        self.wrong_pred_fpga = 0
        self.correct_pred_cpu = 0
        self.wrong_pred_cpu = 0

        pred_fpga = np.argmax(self.act_out_2,axis=1)

        pred_cpu = np.argmax(self.act_out_2_cpu,axis=1)
        tmp = 0
        for i in range(0,self.minibatch_size):
            if pred_fpga[i] == self.ground_truth[i]:
                tmp += 1

        if tmp != np.sum(pred_fpga == self.ground_truth):
            print("=====================\n=====================\n=====================\nNOT EQUAL, BAD counting.=====================\n=====================\n=====================\n")
        self.correct_pred_fpga += np.sum(pred_fpga == self.ground_truth)
        self.wrong_pred_fpga += np.sum(pred_fpga != self.ground_truth)

        self.correct_pred_cpu += np.sum(pred_cpu == self.ground_truth)
        self.wrong_pred_cpu += np.sum(pred_cpu != self.ground_truth)


    def verify_results(self):
        print("Result FPGA:")
        print("Shape FPGA Result: {}".format(self.act_out_2_cpu.shape))
        print(self.act_out_2)

        print("Result CPU:")

        # result[result < 0] = 0
        print("Shape CPU Result: {}".format(self.act_out_2_cpu.shape))
        print(self.act_out_2_cpu)
        do_benchmark = False
        # finally use np.allclose
        if np.array_equal(self.act_out_2_cpu, self.act_out_2):
            print("EXACT FPGA CPU COMPARISON PASSED!!")
        else:
            print("EXACT CPU COMPARISON DID NOT PASS.")
            print("Max Error: {0}".format(np.max(np.abs(self.act_out_2_cpu - self.act_out_2))))
            print("Min Error: {0}".format(np.min(np.abs(self.act_out_2_cpu - self.act_out_2))))

        if np.allclose(self.act_out_2_cpu, self.act_out_2):
            print("FPGA CPU COMPARISON PASSED!!")
            return True
        else:
            print("FPGA CPU COMPARISON DID NOT PASS.")
            print("Max Error: {0}".format(np.max(np.abs(self.act_out_2_cpu - self.act_out_2))))
            print("Min Error: {0}".format(np.min(np.abs(self.act_out_2_cpu - self.act_out_2))))
            return False

    def benchmark(self):

        print("Benchmarking...")
        ops = self.getops()


        start_time = time.time()
        for i in range(self.fpga_iter):
            self.fpga_function(True)
            sys.stdout.write("Run {0}/{1} done.\r".format(i + 1, self.fpga_iter))

        fpga_time = time.time() - start_time

        print("FPGA Time: {0} usec. {1:.3f} GFLOPS".format(fpga_time / self.fpga_iter * 1e6, ops / 1e9 / (fpga_time / self.fpga_iter)))

        print("Running CPU benchmark..")

        start_time = time.time()
        for i in range(self.cpu_iter):
            self.cpu_function(True)
            sys.stdout.write("Run {0}/{1} done.\r".format(i + 1, self.cpu_iter))

        cpu_time = time.time() - start_time

        print("CPU Time: {0} usec. {1:.3f} GFLOPS".format(cpu_time / self.cpu_iter * 1e6, ops / 1e9 / (cpu_time / self.cpu_iter)))

    def print_accuracy(self):
        print("CPU accuracy: {:.2%}".format(self.correct_pred_cpu/(self.correct_pred_cpu+self.wrong_pred_cpu)))
        print("FPGA accuracy: {:.2%}".format(self.correct_pred_fpga / (self.correct_pred_fpga + self.wrong_pred_fpga)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPGA bitstream loading from python")
    parser.add_argument("bitstream", type=argparse.FileType('rb'))

    args = parser.parse_args()
    fpga_class = FPGATest(args.bitstream)
    fpga_class.create_buffers()
    fpga_class.send_buffers_to_device()
    batches = int(fpga_class.x_test.shape[0]/fpga_class.minibatch_size)
    print("Running {} batches...".format(batches))
    for i in range(0, batches):
        fpga_class.set_input(i)
        fpga_class.fpga_function(True)
        fpga_class.retrieve_buffers_from_device()
        fpga_class.cpu_function(True)
        fpga_class.count_accuracy()
        fpga_class.print_accuracy()

    fpga_class.print_accuracy()
    fpga_class.finish_device_queue()
    #if fpga_class.verify_results() and False:
    #    fpga_class.benchmark()
