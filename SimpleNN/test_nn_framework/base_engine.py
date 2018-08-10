import numpy as np
import keras
from keras.datasets import cifar10 as input_data
import pickle
import sys

from . import settings


class BaseEngine:

    def __init__(self, set_verify_config=False):



        self.LAYER_FILENAME = "layer_cifar_{0}.p"
        self.testbatch_size = 10000
        self.minibatch_size = 10000
        self.epochs = 100

        # layer_height = [28 * 28, 512, 512, 10] # MNIST
        self.layer_height = [32 * 32 * 3, 1024, 512, 512, 10]  # CIFAR-10

        self.layers = len(self.layer_height) - 1
        self.hidden_layers = len(self.layer_height) - 2

        self.learn_rate = 0.005
        self.regulation_strength = 0.002

        self.act = {}
        self.bias = {}
        self.weights = {}
        self.dW = {}
        self.delta = {}

        self.ground_truth = None

        if set_verify_config:

            self.LAYER_FILENAME = "verify_{0}.p"
            self.testbatch_size = 32
            self.minibatch_size = 32
            self.epochs = 100

            self.layer_height = [100, 32, 16, 8, 2]  # verify sizes

            self.layers = len(self.layer_height) - 1
            self.hidden_layers = len(self.layer_height) - 2

        if self.layers + 1 != len(self.layer_height):
            print("Bad network config.")
            exit()

        print("Running with {} hidden layers and {} layers total.".format(self.hidden_layers, self.layers))
        if set_verify_config:
            print("Generating fake data...")

            self.y_test = np.zeros((self.minibatch_size, ), dtype=settings.NN_T)
            self.x_test = np.zeros((self.minibatch_size, self.layer_height[0]), dtype=settings.NN_T)

            self.y_train = np.zeros((self.minibatch_size, self.layer_height[self.layers]), dtype=settings.NN_T)
            self.x_train = np.zeros((self.testbatch_size, self.layer_height[self.layers]), dtype=settings.NN_T)

        else:
            print("Loading data...")
            (self.x_train, self.y_train), (self.x_test, self.y_test) = input_data.load_data()

            self.y_train = self.y_train.reshape((self.y_train.shape[0],))
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.layer_height[0])
            self.x_train = self.x_train.astype('float32')
            self.x_train /= 255

            self.y_test = self.y_test.reshape((self.y_test.shape[0],))
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.layer_height[0])
            self.x_test = self.x_test.astype('float32')
            self.x_test /= 255

            self.y_train = keras.utils.to_categorical(self.y_train, self.layer_height[self.layers])
            self.y_test = keras.utils.to_categorical(self.y_test, self.layer_height[self.layers])

        self.correct_pred = 0
        self.wrong_pred = 0

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

    def get_testops(self):
        # Bias additions and broadcast plus matrix mult.
        ops = 0
        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            ops += rows_out * self.testbatch_size + (2 * rows_in - 1) * self.testbatch_size * rows_out

        # Final softmax
        ops += self.layer_height[self.layers] * (4 + 100) + 1
        print("Total test operations: {0} billion per batch.".format(ops / 1e9))
        return ops

    def get_trainops(self):
        # Bias additions and broadcast plus matrix mult.
        ops = 0
        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            ops += rows_out * self.minibatch_size + (2 * rows_in - 1) * self.minibatch_size * rows_out

        # Final softmax
        ops += self.layer_height[self.layers] * (4 + 100) + 1
        print("Total train operations: {0} billion per batch.".format(ops / 1e9))
        return ops

    def create_buffers(self, pretrained=True):
        print("Generating/loading data...")

        ram_act = 0
        ram_weights = 0
        ram_bias = 0

        np.random.seed(0)

        for layer in range(0, self.layers):
            rows_in = self.layer_height[layer]
            rows_out = self.layer_height[layer + 1]
            size_act = (self.minibatch_size, rows_in)
            size_delta = (self.minibatch_size, rows_in)
            size_weights = (rows_in, rows_out)
            size_bias = (1, rows_out)

            # Layer input
            self.act[layer] = self.aligned(np.zeros(size_act, settings.NN_T), alignment=64)

            # Delta
            self.delta[layer] = self.aligned(np.zeros(size_delta, settings.NN_T), alignment=64)

            if pretrained:
                # Load layer file
                layer_file = pickle.load(open(self.LAYER_FILENAME.format(layer), 'rb'))
                # Layer weights
                self.weights[layer] = self.aligned(layer_file['weights'].reshape(size_weights), alignment=64)
                self.bias[layer] = self.aligned(layer_file['bias'].reshape(size_bias), alignment=64)
            else:
                # Layer weights
                self.weights[layer] = self.aligned(np.random.uniform(-0.01,0.01,size=size_weights).astype(dtype=settings.NN_T), alignment=64)
                self.bias[layer] = self.aligned(np.random.uniform(-0.01,0.01,size=size_bias).astype(dtype=settings.NN_T), alignment=64)
            # Temp buffer
            self.dW[layer] = self.aligned(np.zeros(size_weights, dtype=settings.NN_T), alignment=64)
            # Memory
            ram_act += np.size(self.act[layer]) * 4 / 1e6
            ram_weights += np.size(self.weights[layer]) * 4 / 1e6
            ram_bias += np.size(self.bias[layer]) * 4 / 1e6

        final_output_size = (self.minibatch_size, self.layer_height[self.layers])
        # Final output
        self.act[self.layers] = self.aligned(np.zeros(final_output_size, settings.NN_T), alignment=64)

        final_delta_size = (self.minibatch_size, self.layer_height[self.layers])
        # First delta
        self.delta[self.layers] = self.aligned(np.zeros(final_delta_size, settings.NN_T), alignment=64)

        ground_truth_size = (self.minibatch_size, self.layer_height[self.layers])
        # Ground truth
        self.ground_truth = self.aligned(np.zeros(ground_truth_size, settings.NN_T), alignment=64)

        # Memory
        ram_act += np.size(self.act[self.layers]) * 4 / 1e6

        print("Activation total memory: {} MB".format(ram_act))
        print("Weights total memory: {} MB".format(ram_weights))
        print("Bias total memory: {} MB".format(ram_bias))

    def set_input(self, input_values, ground_truth):
        self.act[0] = self.aligned(input_values, alignment=64)

        self.ground_truth = self.aligned(ground_truth, alignment=64)

    def set_train_input(self, batch):
        self.act[0] = self.aligned(self.x_train[batch * self.minibatch_size:(batch + 1) * self.minibatch_size, :],
                                   alignment=64)

        self.ground_truth = self.aligned(self.y_train[batch * self.minibatch_size:(batch + 1) * self.minibatch_size, :],
                                         alignment=64)

    def set_test_input(self, batch):
        self.act[0] = self.aligned(self.x_test[batch * self.testbatch_size:(batch + 1) * self.testbatch_size, :],
                                   alignment=64)

        self.ground_truth = self.aligned(self.y_test[batch * self.testbatch_size:(batch + 1) * self.testbatch_size, :],
                                         alignment=64)

    def reset_accuracy(self):
        self.correct_pred = 0
        self.wrong_pred = 0

    def count_accuracy(self):
        prediction = np.argmax(self.act[self.layers], axis=1)
        ground_truth = np.argmax(self.ground_truth, axis=1)

        self.correct_pred += np.sum(prediction == ground_truth)
        self.wrong_pred += np.sum(prediction != ground_truth)

    def get_act(self, layer=0):
        return self.act[layer]

    def get_weights(self, layer=0):
        return self.weights[layer]

    def get_dW(self, layer=0):
        return self.dW[layer]

    def get_bias(self, layer=0):
        return self.bias[layer]

    def get_delta(self, layer=0):
        return self.delta[layer]

    def verify_act(self, comparative_engine, layer=0):
        return self.verify(comparative_engine.get_act(layer),self.get_act(layer))

    def verify_weights(self, comparative_engine, layer=0):
        return self.verify(comparative_engine.get_weights(layer),self.get_weights(layer))

    def verify_dW(self, comparative_engine, layer=0):
        return self.verify(comparative_engine.get_dW(layer),self.get_dW(layer))

    def verify_bias(self, comparative_engine, layer=0):
        return self.verify(comparative_engine.get_bias(layer),self.get_bias(layer))

    def verify_delta(self, comparative_engine, layer=0):
        return self.verify(comparative_engine.get_delta(layer),self.get_delta(layer))

    @staticmethod
    def verify(comparative, local):
        if np.array_equal(comparative, local):
            print("EXACT COMPARISON PASSED!!")
            return True
        else:
            if np.allclose(comparative, local):
                print("COMPARISON PASSED!!")
                print("Max Error: {0}".format(np.max(np.abs(comparative - local))))
                print("Min Error: {0}".format(np.min(np.abs(comparative - local))))
                print("Tot Error: {0}".format(np.sum(np.abs(comparative - local))))
                return True
            else:
                print("COMPARISON DID NOT PASS.")
                print("Shape Engine Data: {}".format(local.shape))
                print(local)
                print("Shape Comparitive Data: {}".format(comparative.shape))
                print(comparative)
                print("Max Error: {0}".format(np.max(np.abs(comparative - local))))
                print("Min Error: {0}".format(np.min(np.abs(comparative - local))))
                print("Tot Error: {0}".format(np.sum(np.abs(comparative - local))))
                return False

    def print_accuracy(self):
        print("Engine Accuracy: {:.2%}".format(self.correct_pred / (self.correct_pred + self.wrong_pred)))
