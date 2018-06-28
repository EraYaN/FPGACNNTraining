import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt

num_inputs = 784
num_hidden = 1000
num_output = 10
num_passes = 200
num_examples = 32000
num_testexamples = 2500
num_batches = 1000
num_batchsize = int(num_examples / num_batches)
layers = 1 + 2
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


# Helper function to evaluate the total loss on the dataset
def calculate_loss(X, y, W, B):
    a = {}
    z = {}
    a[0] = X
    forward(a, W, B, z)
    probs = a[layers]
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(y.shape[0]), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    sum = 0
    for i in range(1, layers + 1):
        sum += np.sum(np.square(W[i]))
    data_loss += regulation_strength / 2 * sum
    return 1. / num_examples * data_loss

def get_data(num):
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data/')
    return mnist.data, mnist.target
    #return sklearn.datasets.make_blobs(num, n_features=num_inputs, centers=num_output, cluster_std=0.5, random_state=0)

def main_func():
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = get_data(num_examples)
    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

    w = {}
    b = {}
    a = {}
    z = {}
    for i in range(1, layers + 1):
        w[i] = np.random.rand(n[i - 1], n[i]) / np.sqrt(n[i - 1])
        b[i] = np.zeros((1, n[i]))

    train(X, y, a, w, b, z, True)

    print("Plotting results...")
    plot_decision_boundary(X, y, lambda x: decision(predict(x, w, b)))
    plt.title("Decision Boundary for hidden layer size {}".format(num_hidden))

    print("Running tests...")
    X, y = get_data(num_testexamples)
    test_loss = calculate_loss(X, y, w, b)
    print("Test loss: {0:.2f}".format(test_loss/num_testexamples*1e6))

    print("Individual Tests...")
    samples, y_indv = get_data(5)

    for i in range(samples.shape[0]):
        sample = samples[i,:]
        final_prob = test(sample, w, b)
        final = decision(final_prob)
        confidence = final_prob[0][final[0]]
        plt.scatter(sample[0], sample[1], c='black')
        print("Result for {0} => {1} with {2:.1%} confidence. Should be {4}".format(sample, final, confidence, final_prob, y_indv[i]))

    plt.show()


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
    for i in range(1, layers + 1):
        # print("Shape of W[{0}]: {1}".format(i,W[i].shape))
        # print("Shape of b[{0}]: {1}".format(i, B[i].shape))
        # print("Shape of A[{0}]: {1}".format(i-1, A[i-1].shape))
        Z[i] = infer_layer(A[i - 1], W[i], B[i])
        A[i] = activate(Z[i], i)


def backward(A, W, B, probs, y):
    dW = {}
    db = {}
    delta = {layers + 1: probs}
    print(range(num_batchsize))
    delta[layers + 1][range(num_batchsize), y] -= 1
    for i in range(layers, 0, -1):
        # print("Shape of W[{0}]: {1}".format(i,W[i].shape))
        # print("Shape of b[{0}]: {1}".format(i, B[i].shape))
        # print("Shape of A[{0}]: {1}".format(i-1, A[i-1].shape))
        dW[i] = (A[i - 1].T).dot(delta[i + 1]) / A[i - 1].T.shape[0]
        #print("Delta\n")
        # print(delta[i+1])
        db[i] = np.sum(delta[i + 1], axis=0, keepdims=True)
        # print(db[i])
        # print(delta[i-1])
        # print(W[i].T)
        #print((delta[i + 1].dot(W[i].T)))
        #print((1 - np.power(A[i - 1], 2)))
        delta[i] = delta[i + 1].dot(W[i].T) * (1 - np.power(A[i - 1], 2))
        #print(delta[i])
        # print(delta[i])
        # exit(0)

    for i in range(layers, 0, -1):
        dW[i] += regulation_strength * W[i]

    for i in range(1, layers + 1):
        W[i] += -learn_rate * dW[i]
        B[i] += -learn_rate * db[i]


def train(X, y, A, W, B, Z, print_loss=False):
    # Gradient descent. For each batch...
    for i in range(0, num_passes):
        for batch in range(0, num_batches):
            batch_start = batch * num_batchsize
            A[0] = np.array(X[batch_start:batch_start + num_batchsize, :])
            y_batch = np.array(y[batch_start:batch_start + num_batchsize])
            #print(A[0].shape)
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
                print("Loss after iteration {0}, batch {1}: {2:.2f}".format(i, batch, calculate_loss(X, y, W, B)/num_examples*1e6))

    if print_loss:
        print("Loss after last iteration ({0}): {1:.2f}".format(num_passes, calculate_loss(X, y, W, B)/num_examples*1e6))


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