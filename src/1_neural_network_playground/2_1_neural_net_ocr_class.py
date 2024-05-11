import numpy as np
import math
from matplotlib import pyplot as plt

np.random.seed(3)

DATA_DIR = '../../MNIST_DATA'
TEST_DATA_FILENAME = DATA_DIR + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + '/train-images-idx3-ubyte/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

# convert bytes to ints
def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        n_images = bytes_to_int(f.read(4))

        if n_max_images:
            n_images = min(n_images, n_max_images) #prevent silly mistakes
        
        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))

        for image_idx in range(n_images):
            image_data = np.frombuffer(f.read(n_rows*n_cols), dtype=np.uint8)
            image = image_data.reshape((n_rows, n_cols))
            images.append(image.T.flatten())
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        n_labels = bytes_to_int(f.read(4))
        
        if n_max_labels:
            n_labels = min(n_labels, n_max_labels) #avoid silly mistake

        label_data = np.frombuffer(f.read(n_labels), dtype=np.uint8)
        labels = label_data.tolist()
    return labels

class Linear():
    def __init__(self, in_size, out_size):
        # init weights
        self.W = np.random.randn(in_size, out_size) * 0.1
        # init bias
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]
        self.gradW = None
        self.gradB = None
        self.gradInput = None

    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, nextgrad):
        self.gradW = np.dot(self.X.T, nextgrad)
        self.gradB = np.sum(nextgrad, axis=0)
        self.gradInput = np.dot(nextgrad, self.W.T)
        return self.gradInput, [self.gradW, self.gradB]

class ReLU():
    def __init__(self):
        self.params = []
        self.gradInput = None

    def forward(self, X):
        self.output = np.maximum(X, 0)
        return self.output

    def backward(self, nextgrad):
        self.gradInput = nextgrad.copy()
        self.gradInput[self.output <=0] = 0
        return self.gradInput, []

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return out

class CrossEntropy():
    def forward(self, X, y):
        self.m = y.shape[0]
        self.p = softmax(X)
        cross_entropy = -np.log(self.p[range(self.m), y])
        loss = cross_entropy[0] / self.m
        return loss

    def backward(self, X, y):
        y_idx = y.argmax()
        grad = softmax(X)
        grad[range(self.m), y] -= 1
        grad /= self.m
        return grad

class NN():
    def __init__(self, lossfunc=CrossEntropy()):
        self.params = []
        self.layers = []
        self.loss_func = lossfunc
        self.grads = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.append(layer.params)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, nextgrad):
        self.clear_grad_param()
        for layer in reversed(self.layers):
            nextgrad, grad = layer.backward(nextgrad)
            self.grads.append(grad)
        return self.grads

    def train_step(self, X, y):
        out = self.forward(X)
        loss = self.loss_func.forward(out, y)
        nextgrad = self.loss_func.backward(out, y)
        l2 = self.backward(nextgrad)
        return loss, l2

    def predict(self, X):
        X = self.forward(X)
        return np.argmax(X, axis=1)

    def predict_scores(self, X):
        X = self.forward(X)
        return X

    def clear_grad_param(self):
        self.grads = []

def update_params(velocity, params, grads, learning_rate=0.01, mu=0.9):
    for v, p, g in zip(velocity, params, reversed(grads)):
        for i in range(len(g)):
            v[i] = mu * v[i] + learning_rate * g[i]
            p[i] -= v[i]
            print("Max Gradient Value: ", np.amax(v[i]))
            print("Gradient Shape: ", v[i].shape)

def minibatch(X, y, minibatch_size):
    n = X.shape[0]
    minibatches = []
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    for i in range(0, n, minibatch_size):
        X_batch = X[i:i + minibatch_size, :]
        y_batch = y[i:i + minibatch_size]
        minibatches.append((X_batch, y_batch))

    return minibatches

def train(net, X_train, y_train, minibatch_size, epoch, learning_rate, mu=0.9, X_val=None, y_val=None):
    val_loss_epoch = []
    minibatches = minibatch(X_train, y_train, minibatch_size)
    minibatches_val = minibatch(X_val, y_val, minibatch_size)

    return 0

    for i in range(epoch):
        loss_batch = []
        val_loss_batch = []
        velocity = []

        for param_layer in net.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        for X_mini, y_mini in minibatches:
            loss, grads = net.train_step(X_mini, y_mini)
            loss_batch.append(loss)
            update_params(velocity, net.params, grads, learning_rate=learning_rate, mu=mu)

        for X_mini_val, y_mini_val in minibatches_val:
            val_loss, _ = net.train_step(X_mini, y_mini)
            val_loss_batch.append(val_loss)

        m_train = X_train.shape[0]
        m_val = X_val.shape[0]
        y_train_pred = np.array([], dtype="int64")
        y_val_pred = np.array([], dtype="int64")
        y_train1 = []
        y_vall = []

        for i in range(0, m_train, minibatch_size):
            X_tr = X_train[i:i + minibatch_size, : ]
            y_tr = y_train[i:i + minibatch_size,]
            y_train1 = np.append(y_train1, y_tr)
            y_train_pred = np.append(y_train_pred, net.predict(X_tr))

        for i in range(0, m_val, minibatch_size):
            X_va = X_val[i:i + minibatch_size, : ]
            y_va = y_val[i:i + minibatch_size,]
            y_vall = np.append(y_vall, y_va)
            y_val_pred = np.append(y_val_pred, net.predict(X_va))
            
        train_acc = check_accuracy(y_train1, y_train_pred)
        val_acc = check_accuracy(y_vall, y_val_pred)

        mean_train_loss = sum(loss_batch) / float(len(loss_batch))
        mean_val_loss = sum(val_loss_batch) / float(len(val_loss_batch))
        
        val_loss_epoch.append(mean_val_loss)
        print("Loss = {0} | Training Accuracy = {1} | Val Loss = {2} | Val Accuracy = {3}".format(mean_train_loss, train_acc, mean_val_loss, val_acc))
    return net

def check_accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def main():
    X_train = read_images(TRAIN_DATA_FILENAME) 
    y_train = read_labels(TRAIN_LABELS_FILENAME) 
    X_test = read_images(TEST_DATA_FILENAME)
    y_test = read_labels(TEST_LABELS_FILENAME)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Normalize the pixel data from 0-255 to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    input_dim = X_train.shape[1]

    iteration = 10
    learning_rate = 0.1
    hidden_nodes = 64
    output_nodes = 10

    nn = NN()
    nn.add_layer(Linear(input_dim, hidden_nodes))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(hidden_nodes, hidden_nodes))
    nn.add_layer(ReLU())
    nn.add_layer(Linear(hidden_nodes, output_nodes))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # nn = train(nn, X_train, y_train, minibatch_size=200, epoch=10, learning_rate=learning_rate, X_val=X_test, y_val=y_test)

if __name__ == '__main__':
    main()