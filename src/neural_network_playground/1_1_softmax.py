import numpy as np

import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = prob

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
act1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
act2 = Activation_Softmax()

dense1.forward(X)
act1.forward(dense1.output)

dense2.forward(act1.output)
act2.forward(dense2.output)

print(act2.output[:5])
