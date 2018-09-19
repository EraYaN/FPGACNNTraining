import numpy as np
import time

from . import settings
from .base_engine import BaseEngine

import sys


class CPUEngine(BaseEngine):
    def __init__(self, set_verify_config=False):
        super().__init__(set_verify_config=set_verify_config)

    def forward(self, act, weights, bias):
        return act.dot(weights) + bias

    def relu(self, act, derivative=False):
        return (act > 0).astype(settings.NN_T) if derivative else np.clip(act,0,None)

    def sigmoid(self, act, derivative=False):
        return act * (1 - act) if derivative else 1 / (1 + np.exp(-act))

    def softmax(self, act):
        e_act = np.exp(act - np.max(act, axis=1, keepdims=True))
        divisor = np.sum(e_act, axis=1, keepdims=True)
        return np.divide(e_act, divisor)

    def first_delta(self, predictions, ground_truth):
        m = ground_truth.shape[0]
        return (predictions - ground_truth)/m

    def cross_entropy(self, predictions, ground_truth, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = np.sum(np.sum(-ground_truth * np.log(predictions + 1e-9))) / N
        return ce

    def backward(self, act, weights, bias, delta_next, learn_rate, layer):
        #print("Start layer {}".format(layer))
        dW = act.T.dot(delta_next)

        #print("shape act{}".format(layer), act.shape)
        #print("shape delta{}".format(layer+1), delta_next.shape)
        #print("shape dW{}".format(layer),dW.shape)
        #print("shape weights{}".format(layer), weights.shape)

        db = np.sum(delta_next, axis=0, keepdims=True)

        if layer > 0:
            delta = delta_next.dot(weights.T)

            # apply relu derive product
            delta *= self.relu(act, derivative=True)
        else:
            delta = None

        #print(delta)

        #print("shape delta{}".format(layer), delta.shape)

        #dW += self.regulation_strength * weights

        weights -= learn_rate * dW
        bias -= learn_rate * db

        return weights, bias, delta, dW

    def bw_function(self, benchmark=False):

        for layer in range(self.layers, -1, -1):
            #print("backward layer {}".format(layer))
            is_last_layer = layer == self.layers

            if is_last_layer:
                self.delta[layer] = self.first_delta(self.act[layer], self.ground_truth)
            else:
                if np.sum(self.act[layer]) == 0:
                    print("\nact{}".format(layer-1), np.sum(self.act[layer-1]))
                    print("weight{} max min".format(layer - 1), np.max(self.weights[layer - 1]), np.min(self.weights[layer - 1]))
                    print("bias{} max min".format(layer - 1), np.max(self.bias[layer - 1]),
                          np.min(self.bias[layer - 1]))

                    print("act{} recalc".format(layer), np.sum(self.relu(self.forward(self.act[layer-1],self.weights[layer - 1],self.bias[layer - 1]))))
                    print("act{} is zero".format(layer), np.sum(self.act[layer]))
                self.weights[layer], self.bias[layer], self.delta[layer], self.dW[layer] = self.backward(
                    self.act[layer],
                    self.weights[layer],
                    self.bias[layer],
                    self.delta[layer+1],
                    self.learn_rate,
                    layer)

    def fw_function(self, benchmark=False):
        for layer in range(0, self.layers):
            #print("Forward layer {}".format(layer))
            self.act[layer + 1] = self.forward(self.act[layer], self.weights[layer], self.bias[layer])
            
            if layer + 1 != self.layers:
                self.act[layer + 1] = self.relu(self.act[layer + 1])
            else:
                self.act[layer + 1] = self.softmax(self.act[layer + 1])

    def train(self):
        batches = int(self.x_train.shape[0] / self.minibatch_size)
        if self.batch_limit is not None:
            batches = self.batch_limit
        #batches = 20
        print("Training with {} batches of size {}".format(batches, self.minibatch_size))
        for epoch in range(0, self.epochs):
            print("Epoch {} of {}...".format(epoch+1, self.epochs))
            #self.shuffle_train_set()
            for batch in range(0, batches):
                self.set_train_input(batch)
                #print("input",self.act[0][0])

                self.fw_function()
                #print("after fw act2 nz", np.count_nonzero(self.act[2]))
                self.bw_function()
                sys.stdout.write("Batch {} of {} complete.\r".format(batch+1, batches))
            print("Epoch {} of {} is complete.".format(epoch + 1, self.epochs))
            self.test()

    def test(self):
        batches = int(self.x_test.shape[0] / self.testbatch_size)

        print("Running {} batches on CPU...".format(batches))
        total_time = 0.0
        self.reset_accuracy()
        for i in range(0, batches):
            self.set_test_input(i)
            start_time = time.time()

            self.fw_function(True)

            total_time += time.time() - start_time
            self.count_accuracy()
            #if i % 10 == 0:
            #    print("CPU Test Batch {}/{} done.".format(i + 1, batches))

        self.loss = self.cross_entropy(self.act[self.layers], self.ground_truth)
        self.print_accuracy()
        return total_time
