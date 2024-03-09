from MNIST_data import generate_MNIST_data

import numpy as np
import math

class NN():
    def __init__(self, loss_func, layer_config, iteration, learning_rate, minibatch_size, epoch, mu):
        # Init Data
        self.data = generate_MNIST_data(1000, 1000)
        self.X_train = self.process_data()
        self.y_train = np.array(self.data[1])
        print(self.y_train.shape)

        # Init Loss Function
        self.sup_loss = ["CrossEntropy"] 
        self.loss_func = loss_func
        self.check_supported_loss()
        self.loss_func = self.get_loss_func()

        self.input_dim = self.X_train.shape[1]
        self.output_nodes = 10
        self.X_n = self.X_train.shape[0]

        # Init Hyperparameters
        self.iteration = iteration
        self.minibatch_size = minibatch_size
        self.epoch = epoch
        self.mu = mu

        self.params = []
        self.layer_config = layer_config
        self.layers = []
        self.grads = []

    def process_data(self):
        X = np.array(self.data[0])
        X = X / 255.0
        return X

    def check_supported_loss(self):
        if self.loss_func not in self.sup_loss:
            raise ValueError("Unsupported feature extraction method. Supported methods are: {}".format(self.sup_loss))

    def get_loss_func(self):
        if self.loss_func == "CrossEntropy":
            return CrossEntropy()

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

    def clear_grad_param(self):
        self.grads = []

    def update_params(self, velocity, grads):
        for v, p, g in zip(velocity, self.params, reversed(grads)):
            for i in range(len(g)):
                v[i] = self.mu * v[i] + self.learning_rate * g[i]
                p[i] -= v[i]
                print("Max Gradient Value: ", np.amax(v[i]))
                print("Gradient Shape: ", v[i].shape)

    def minibatch(self):
        minibatches = []
        permutation = np.random.permutation(self.X_n)
        X = self.X_train[permutation]
        y = self.y_train[permutation]

        for i in range(0, self.X_n, self.minibatch_size):
            X_batch = X[i:i + self.minibatch_size, :]
            y_batch = y[i:i + self.minibatch_size]
            minibatches.append((X_batch, y_batch))

        return minibatches

    def initilize_layers(self):
        self.add_layer(Linear(self.input_dim, 64))
        self.add_layer(ReLU())
        self.add_layer(Linear(64, self.output_nodes))
        # if len(self.layer_config) < 1:
        #     raise ValueError("NN must have at least one hidden layer")

        # self.add_layer(Linear(self.input_dim, self.layer_config[0]))
        # self.add_layer(ReLU())

        # # If there is only one hidden layer, add the last linear layer directly
        # if len(self.layer_config) == 1:
        #     self.add_layer(Linear(self.layer_config[0], self.output_nodes))
        # else:
        #     # Iterate through the remaining layer_config elements
        #     for i in range(1, len(self.layer_config)):
        #         # Add a linear layer
        #         self.add_layer(Linear(self.layer_config[i - 1], self.layer_config[i]))
        #         # Add ReLU activation function after each linear layer
        #         self.add_layer(ReLU())

        #     # Add the last linear layer connecting to the output nodes
        #     self.add_layer(Linear(self.layer_config[-1], self.output_nodes))

    def train(self):
        # val_loss_epoch = []
        minibatches = self.minibatch()
        # minibatches_val = self.minibatch(X_val, y_val, minibatch_size)

        for i in range(self.epoch):
            loss_batch = []
            # val_loss_batch = []
            velocity = []

            for param_layer in self.params:
                p = [np.zeros_like(param) for param in list(param_layer)]
                velocity.append(p)

            for X_mini, y_mini in minibatches:
                loss, grads = self.train_step(X_mini, y_mini)
                loss_batch.append(loss)
                self.update_params(velocity, grads)

            # for X_mini_val, y_mini_val in minibatches_val:
            #     val_loss, _ = net.train_step(X_mini, y_mini)
            #     val_loss_batch.append(val_loss)

            # m_val = X_val.shape[0]
            y_train_pred = np.array([], dtype="int64")
            # y_val_pred = np.array([], dtype="int64")
            y_train1 = []
            # y_vall = []

            for i in range(0, self.X_n, self.minibatch_size):
                X_tr = self.X_train[i:i + self.minibatch_size, : ]
                y_tr = self.y_train[i:i + self.minibatch_size,]
                y_train1 = np.append(y_train1, y_tr)
                y_train_pred = np.append(y_train_pred, self.predict(X_tr))

            # for i in range(0, m_val, minibatch_size):
            #     X_va = X_val[i:i + minibatch_size, : ]
            #     y_va = y_val[i:i + minibatch_size,]
            #     y_vall = np.append(y_vall, y_va)
            #     y_val_pred = np.append(y_val_pred, net.predict(X_va))
                
            train_acc = self.check_accuracy(y_train1, y_train_pred)
            # val_acc = self.check_accuracy(y_vall, y_val_pred)

            mean_train_loss = sum(loss_batch) / float(len(loss_batch))
            # mean_val_loss = sum(val_loss_batch) / float(len(val_loss_batch))
            
            # val_loss_epoch.append(mean_val_loss)
            print("Loss = {0} | Training Accuracy = {1} | Val Loss = {2} | Val Accuracy = {3}".format(mean_train_loss, train_acc))
        return net

    def check_accuracy(self, y_true, y_pred):
        return np.mean(y_pred == y_true)

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

class CrossEntropy():
    def forward(self, X, y):
        self.m = y.shape[0]
        self.p = self.softmax(X)
        cross_entropy = -np.log(self.p[range(self.m), y])
        loss = cross_entropy[0] / self.m
        return loss

    def backward(self, X, y):
        y_idx = y.argmax()
        grad = self.softmax(X)
        grad[range(self.m), y] -= 1
        grad /= self.m
        return grad

    def softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return out
