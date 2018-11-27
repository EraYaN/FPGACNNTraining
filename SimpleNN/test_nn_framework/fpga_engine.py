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

from . import settings
from .base_engine import BaseEngine


class FPGAEngine(BaseEngine):

    def __init__(self, kernel_file, set_verify_config=False):
        super().__init__(set_verify_config=set_verify_config)

        assert self.minibatch_size == self.testbatch_size, "minibatch size and testbatch size need to be the same for the FPGA engine."

        self.act_buffers = {}
        self.delta_buffers = {}
        self.bias_buffers = {}
        self.weights_buffers = {}
        self.dW_buffers = {}

        self.weightsT = {}
        self.dWT = {}

        self.ground_truth_buffer = None

        print("OpenCL Version v{}".format(".".join([str(i) for i in cl.get_cl_header_version()])))
        print("Finding platform....")
        platform = self.findPlatform(settings.VENDOR_NAME)
        if not platform:
            print("ERROR: Platform not found for name {0}".format(settings.VENDOR_NAME))
            exit(1)

        print("Getting devices...")
        devices = platform.get_devices(device_type=settings.DEVICE_TYPE)
        if len(devices) < 1:
            print("ERROR: No device found for type {0}.".format(settings.DEVICE_TYPE))
            exit(1)

        devices = [devices[0]]

        self.ctx = cl.Context(devices=devices)

        if settings.DEVICE_TYPE == cl.device_type.ACCELERATOR:
            print("Reading binary...")
            binary = kernel_file.read()

            binaries = [binary] * len(devices)

            program = cl.Program(self.ctx, devices, binaries)
        else:
            print("Reading program...")
            binary = kernel_file.read()

            print("Building...")
            program = cl.Program(self.ctx, binary.decode('utf-8')).build()

        self.kForward = program.forward
        self.kForwardSoftMax = program.forward_softmax
        self.kBackwardFirstDelta = program.backward_first_delta
        self.kBackward = program.backward

        self.kForward.set_scalar_arg_dtypes([None, None, None, None, np.int32, np.int32, np.int32, np.int32])
        self.kForwardSoftMax.set_scalar_arg_dtypes([None, np.int32, np.int32])
        self.kBackwardFirstDelta.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32])
        self.kBackward.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, settings.NN_T, np.int32, np.int32, np.int32, np.int32])

        self.queue = cl.CommandQueue(self.ctx)

    def findPlatform(self, strname):
        platforms = cl.get_platforms()
        print(platforms)
        for platform in platforms:
            if settings.VENDOR_NAME in platform.get_info(cl.platform_info.NAME).lower():
                return platform
        return None

    def cross_entropy(self, predictions, ground_truth, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = np.sum(np.sum(-ground_truth * np.log(predictions + 1e-9))) / N
        return ce

    def fw_function(self, benchmark=False):
        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            is_last_layer = layer + 1 == self.layers

            #print("Running layer {} ({}->{}) kernel, with relu: {}".format(layer, rows_in, rows_out,
                                                                           #1 if not is_last_layer else 0))
            self.kForward.set_args(self.act_buffers[layer], self.weights_buffers[layer], self.bias_buffers[layer],
                                   self.act_buffers[layer + 1],
                                   self.minibatch_size, rows_in, rows_out, 1 if not is_last_layer else 0)
            cl.enqueue_task(self.queue, self.kForward).wait()

            if is_last_layer:
                #print("Running layer {} softmax kernel".format(layer))
                self.kForwardSoftMax.set_args(self.act_buffers[layer + 1],
                                              self.minibatch_size, rows_out)
                cl.enqueue_task(self.queue, self.kForwardSoftMax).wait()
                #print("Running layer {} softmax kernel. COMPLETED.".format(layer))

    def bw_function(self, benchmark=False):
        for layer in range(self.layers, -1, -1):
            rows_in = self.layer_height[layer]
            is_last_layer = layer == self.layers
            if is_last_layer:
                #print("Running layer {} first_delta kernel".format(layer))
                self.kBackwardFirstDelta.set_args(self.act_buffers[layer], self.ground_truth_buffer,
                                                  self.delta_buffers[layer], self.minibatch_size,
                                                  self.layer_height[layer])
                cl.enqueue_task(self.queue, self.kBackwardFirstDelta).wait()
                #print("Running layer {} first_delta kernel. COMPLETED".format(layer))
            else:
                rows_out = self.layer_height[layer + 1]

                #print("Running layer {} backwards kernel".format(layer))
                self.kBackward.set_args(self.act_buffers[layer], self.weights_buffers[layer], self.dW_buffers[layer],
                                        self.bias_buffers[layer], self.delta_buffers[layer],
                                        self.delta_buffers[layer + 1],
                                        self.learn_rate, self.minibatch_size, rows_in, rows_out, layer)
                cl.enqueue_task(self.queue, self.kBackward).wait()

                #print("Running layer {} backwards kernel. COMPLETED".format(layer))

    def train(self):
        batches = int(self.x_train.shape[0] / self.minibatch_size)
        if self.batch_limit is not None:
            batches = self.batch_limit
        # batches = 20
        print("Training with {} batches of size {}".format(batches, self.minibatch_size))
        fpga_time = 0.0

        for epoch in range(0, self.epochs):
            print("Epoch {} of {}...".format(epoch + 1, self.epochs))
            #self.shuffle_train_set()
            for batch in range(0, batches):
                self.set_train_input(batch)

                start_time = time.time()
                self.fw_function()

                self.bw_function()
                fpga_time += time.time() - start_time

                sys.stdout.write("Batch {} of {} complete.\r".format(batch + 1, batches))
            print("Epoch {} of {} is complete.".format(epoch + 1, self.epochs))
            self.test()

        return fpga_time

    def test(self):
        batches = int(self.x_test.shape[0] / self.testbatch_size)
        if self.batch_limit is not None:
            batches = self.batch_limit
        print("Running {} batches on FPGA...".format(batches))
        fpga_time = 0.0

        self.reset_accuracy()
        for i in range(0, batches):
            self.set_test_input(i)
            start_time = time.time()

            self.fw_function(True)

            fpga_time += time.time() - start_time
            self.retrieve_buffers_from_device()

            self.count_accuracy()
        self.loss = self.cross_entropy(self.act[self.layers], self.ground_truth)
        self.print_accuracy()

        return fpga_time

    def create_buffers(self, pretrained=True):
        super().create_buffers(pretrained=pretrained)

        self.send_buffers_to_device()

    def set_input(self, input_values, ground_truth):
        super(FPGAEngine, self).set_input(input_values, ground_truth)
        cl.enqueue_copy(self.queue, self.act_buffers[0], self.act[0]).wait()
        cl.enqueue_copy(self.queue, self.ground_truth_buffer, self.ground_truth).wait()

    def set_train_input(self, batch):
        super(FPGAEngine, self).set_train_input(batch)
        cl.enqueue_copy(self.queue, self.act_buffers[0], self.act[0]).wait()
        cl.enqueue_copy(self.queue, self.ground_truth_buffer, self.ground_truth).wait()

    def set_test_input(self, batch):
        super(FPGAEngine, self).set_test_input(batch)
        cl.enqueue_copy(self.queue, self.act_buffers[0], self.act[0]).wait()
        cl.enqueue_copy(self.queue, self.ground_truth_buffer, self.ground_truth).wait()

    def send_buffers_to_device(self):
        mf = cl.mem_flags

        #print("Transferring all buffers to device...")

        for layer in range(0, self.layers):
            self.act_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.act[layer])
            self.delta_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.delta[layer])
            self.bias_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.bias[layer])
            self.weightsT[layer] = self.aligned(self.weights[layer].T.copy(), alignment=64)
            self.weights_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.weightsT[layer])
            self.dWT[layer] = self.aligned(self.dW[layer].T.copy(), alignment=64)
            self.dW_buffers[layer] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.dWT[layer])

        # Final output
        self.act_buffers[self.layers] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.act[self.layers])
        # Final delta
        self.delta_buffers[self.layers] = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.delta[self.layers])
        # Ground truth
        self.ground_truth_buffer = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=self.ground_truth)

    def retrieve_buffers_from_device(self, benchmark=False, all=False):
        if all:
            for layer in range(0, self.layers):
                #if not benchmark:
                    #print("Transferring all values from device for layer {}...".format(layer))
                    # print(self.bias[layer],self.bias_buffers[layer])
                cl.enqueue_copy(self.queue, self.weightsT[layer], self.weights_buffers[layer]).wait()
                self.weights[layer] = self.weightsT[layer].T
                cl.enqueue_copy(self.queue, self.dWT[layer], self.dW_buffers[layer]).wait()
                self.dW[layer] = self.dWT[layer].T
                cl.enqueue_copy(self.queue, self.bias[layer], self.bias_buffers[layer]).wait()
                cl.enqueue_copy(self.queue, self.delta[layer], self.delta_buffers[layer]).wait()
                cl.enqueue_copy(self.queue, self.act[layer], self.act_buffers[layer]).wait()

            cl.enqueue_copy(self.queue, self.delta[self.layers], self.delta_buffers[self.layers]).wait()

        #if not benchmark:
            #print("Transferring final result from device...")
        cl.enqueue_copy(self.queue, self.act[self.layers], self.act_buffers[self.layers]).wait()

    def finish_device_queue(self):
        print("Finishing queue..")
        self.queue.finish()
