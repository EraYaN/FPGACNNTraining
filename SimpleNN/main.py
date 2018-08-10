import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

num_inputs = 64
num_hidden = 32
num_output = 10
num_passes = 100
num_examples = 32000
num_testexamples = 1000
num_batches = 1000
num_batchsize = int(num_examples / num_batches)
hidden_layers = 2
layers = 1 + hidden_layers
n_l = [num_inputs] + [num_hidden] * (layers - 1) + [num_output]
n = tuple(n_l)

learn_rate = 0.001
regulation_strength = 0.001


def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def count_accuracy(X, y, W, B):
    a = {}
    z = {}
    a[0] = X
    forward(a, W, B, z)
    probs = a[layers]

    prediction_cpu = np.argmax(probs, axis=1)

    return np.sum(prediction_cpu == y) / y.shape[0]
    # wrong_pred_cpu = np.sum(prediction_cpu != y)


def make_data(num):
    # mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
    return sklearn.datasets.make_blobs(num, n_features=num_inputs, centers=num_output, cluster_std=0.5, random_state=0)


def main_func():
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = make_data(num_examples)
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

    w = {}
    b = {}
    a = {}
    z = {}
    for i in range(0, layers):
        w[i] = np.random.rand(n[i], n[i + 1]) / np.sqrt(n[i])
        b[i] = np.zeros((1, n[i + 1]))

    train(X, y, a, w, b, z, True)

    # print("Plotting results...")
    # plot_decision_boundary(X, y, lambda x: decision(predict(x, w, b)))
    # plt.title("Decision Boundary for hidden layer size {}".format(num_hidden))

    print("Running tests...")
    X, y = make_data(num_testexamples)
    test_acc = count_accuracy(X, y, w, b)
    print("Test accuracy: {0:.2f}".format(test_acc))

    # print("Individual Tests...")
    # samples, y_indv = make_data(5)
    #
    # for i in range(samples.shape[0]):
    #     sample = samples[i,:]
    #     final_prob = test(sample, w, b)
    #     final = decision(final_prob)
    #     confidence = final_prob[0][final[0]]
    #     plt.scatter(sample[0], sample[1], c='black')
    #     print("Result for {0} => {1} with {2:.1%} confidence. Should be {4}".format(sample, final, confidence, final_prob, y_indv[i]))

    # plt.show()


def test(sample, W, B):
    a = {}
    z = {}
    a[0] = sample.reshape((1, n[0]))
    forward(a, W, B, z)

    return a[layers]


def predict(X, W, B):
    a = {}
    z = {}
    a[0] = X
    forward(a, W, B, z)

    return a[layers]


def decision(result):
    return np.argmax(result, axis=1)


def forward(A, W, B, Z):
    for i in range(0, layers):
        # print("Shape of W[{0}]: {1}".format(i,W[i].shape))
        # print("Shape of b[{0}]: {1}".format(i, B[i].shape))
        # print("Shape of A[{0}]: {1}".format(i-1, A[i-1].shape))
        Z[i + 1] = infer_layer(A[i], W[i], B[i])
        A[i + 1] = activate(Z[i + 1], i)


def backward(A, W, B, probs, y):
    dW = {}
    db = {}
    delta = {layers: probs}
    delta[layers][range(num_batchsize), y] -= 1
    # print("W: ",W)
    # print("A: ",A)
    # print("B: ",B)
    # print("delta: ",delta)
    for i in range(layers - 1, 0, -1):
        # print("Layer {0} out of {1}\n".format(i+1,layers))
        # print("Shape of W[{0}]: {1}".format(i, W[i].shape))
        # print("Shape of b[{0}]: {1}".format(i, B[i].shape))
        # print("Shape of A[{0}]: {1}".format(i, A[i].shape))

        dW[i] = A[i].T.dot(delta[i + 1])
        #
        db[i] = np.sum(delta[i + 1], axis=0, keepdims=True)
        #
        delta[i] = delta[i + 1].dot(W[i].T) * np.clip(A[i], 0, np.Inf)

        #
        # print(delta[i].shape, A[i].shape)
        # # apply relu derive product
        # #delta[i][A[i] < 0] = 0
        #
        W[i] -= learn_rate * dW[i]
        B[i] -= learn_rate * db[i]

        # print("Shape of W[{0}]: {1}".format(i,W[i].shape))
        # print("Shape of b[{0}]: {1}".format(i, B[i].shape))
        # print("Shape of A[{0}]: {1}".format(i-1, A[i-1].shape))
        # dW[i] = (A[i - 1].T).dot(delta[i + 1]) / A[i - 1].shape[0]
        # print("Delta\n")
        # print(delta[i+1])
        # db[i] = np.sum(delta[i + 1], axis=0, keepdims=True)
        # print(db[i])
        # print(delta[i-1])
        # print(W[i].T)
        # print((delta[i + 1].dot(W[i].T)))
        # print((1 - np.power(A[i - 1], 2)))
        # delta[i] = delta[i + 1].dot(W[i].T) * (1 - np.power(A[i - 1], 2))
        # print(delta[i])
        # print(delta[i])
        # exit(0)

    # for i in range(layers, 0, -1):
    #     dW[i] += regulation_strength * W[i]
    #
    # for i in range(1, layers + 1):
    #     W[i] += -learn_rate * dW[i]
    #     B[i] += -learn_rate * db[i]


def train(X, y, A, W, B, Z, print_loss=False):
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        for batch in range(0, num_batches):
            batch_start = batch * num_batchsize
            A[0] = np.array(X[batch_start:batch_start + num_batchsize, :])
            y_batch = np.array(y[batch_start:batch_start + num_batchsize])
            # print(A[0].shape)
            # Forward propagation
            forward(A, W, B, Z)

            probs = A[layers]

            # Backpropagation
            backward(A, W, B, probs, y_batch)

            # Assign new parameters to the model
            # model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and (i * num_batches + batch) % 1000 == 0:
                print("Accuracy after iteration {0}, batch {1}: {2:.2f}".format(i, batch, count_accuracy(X, y, W, B)))

    if print_loss:
        print("Accuracy after last iteration ({0}): {1:.2f}".format(num_passes, count_accuracy(X, y, W, B)))


def infer_layer(X, W, B):
    Z = X.dot(W) + B
    return Z


def activate(Z, i):
    A = np.tanh(Z)

    if (i < layers):
        A = np.tanh(Z)
    else:
        exp_probs = np.exp(Z)
        sum = np.sum(exp_probs, axis=1, keepdims=True)
        A = exp_probs / sum
    # A[A <= 0] = 0
    return A


def cost():
    return 0


if __name__ == "__main__":
    print("Starting...")
    main_func()
    print("Done.")
