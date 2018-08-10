import numpy as np
import time

from . import settings
from .base_engine import BaseEngine


class CPUEngine(BaseEngine):
    def __init__(self, set_verify_config=False):
        super().__init__(set_verify_config=set_verify_config)

    def forward(self, act, weights, bias):
        return act.dot(weights) + bias

    def relu(self, act):
        act[act < 0] = settings.NN_T(0)
        return act

    def softmax(self, act):
        # print(act)
        # print(np.max(act, axis=1, keepdims=True))
        e_act = np.exp(act - np.max(act, axis=1, keepdims=True))
        print("softmax", np.divide(e_act, np.sum(e_act, axis=1, keepdims=True)), "e_act", e_act, "npsum e_act",
              np.sum(e_act, axis=1, keepdims=True))
        return np.divide(e_act, np.sum(e_act, axis=1, keepdims=True))

    def first_delta(self, probs, ground_truth):
        print("first delta", -ground_truth, " divided by ", probs + 0.0001)
        return np.divide(-ground_truth, probs + 0.0001)

    def backward(self, act, weights, bias, delta_next, learn_rate):

        dW = act.T.dot(delta_next)

        db = np.sum(delta_next, axis=0, keepdims=True)

        delta = delta_next.dot(weights.T)

        print("shapes:",act.shape,delta_next.shape,dW.shape,weights.shape)

        # apply relu derive product
        delta = np.sign(delta)
        delta[delta == -1] = 0
        print("weights", weights)
        weights -= learn_rate * dW
        print("dW", dW)
        # print("delta_NEXT", delta_next)
        # print("db", learn_rate * db)
        # print("bias", bias)
        bias -= learn_rate * db
        return weights, bias, delta, dW

    def bw_function(self, benchmark=False):
        for layer in range(self.layers, -1, -1):
            print("backward layer {}".format(layer))
            is_last_layer = layer == self.layers

            if is_last_layer:
                self.delta[layer] = self.first_delta(self.act[layer], self.ground_truth)
            else:
                self.weights[layer], self.bias[layer], self.delta[layer], self.dW[layer] = self.backward(self.act[layer],
                                                                                         self.weights[layer],
                                                                                         self.bias[layer],
                                                                                         self.delta[layer + 1],
                                                                                         self.learn_rate)

    def fw_function(self, benchmark=False):
        for layer in range(0, self.layers):
            self.act[layer + 1] = self.forward(self.act[layer], self.weights[layer], self.bias[layer])
            if layer + 1 != self.layers:
                self.act[layer + 1] = self.relu(self.act[layer + 1])
            else:
                self.act[layer + 1] = self.softmax(self.act[layer + 1])

    def train(self):
        batches = int(self.x_train.shape[0] / self.minibatch_size)

        for epoch in range(0, self.epochs):
            # print("Running {} batches...".format(batches))
            for batch in range(0, batches):
                self.set_train_input(batch)

                self.fw_function()

                self.bw_function()

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
            if i % 10 == 0:
                print("CPU Batch {}/{} done.\n".format(i + 1, batches))
                self.print_accuracy()

        return total_time
