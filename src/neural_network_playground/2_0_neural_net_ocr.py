import numpy as np
import math
from matplotlib import pyplot as plt

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

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10,1) -0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,1) -0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def der_ReLU(Z):
    return Z > 0

def softmax(Z):
    Z -= np.max(Z, axis=0)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def forward_prop(W1, b1, W2, b2, X):
    # print("w1; ", W1.shape)
    # print("b1; ", b1.shape)
    # print("w2; ", W2.shape)
    # print("b2; ", b2.shape)

    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# def one_hot(Y, n_class):
#     one_hot_Y = np.zeros((len(Y), n_class))
#     one_hot_Y[np.arange(len(Y)), Y] = 1
#     return one_hot_Y.T

def one_hot(Y, n_classes):
    one_hot_Y = np.zeros((n_classes, len(Y)))
    for i, y in enumerate(Y):
        one_hot_Y[y, i] = 1
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W2, X, one_hot_Y):
    print("Z1: ", Z1)
    print("A1: ", A1)
    print("Z2: ", Z2)
    print("A2: ", A2)
    print("W2: ", W2)

    m = len(one_hot_Y)
    
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1)

    dZ1 = W2.T.dot(dZ2) * der_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

# def get_predictions(A2):
#     return np.argmax(A2, 0)

def get_accuracy_batch(pred, Y):
    pred_labels = np.argmax(pred, axis=0)
    accuracy = np.sum(pred_labels == Y) / Y.shape[0]
    return accuracy

def grad_descent(X_train, y_train, iterations, alpha, batch_size):
    W1, b1, W2, b2 = init_params()
    num_batches = X_train.shape[1] 

    for i in range(iterations):
        print("iteration : ", i)
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch = X_train[:, start:end]
            y_batch = y_train[start:end]

            if X_batch.shape[1] == 0:  # Skip empty batches
                continue
            one_hot_Y = one_hot(y_batch, 10)
            Z1_batch, A1_batch, Z2_batch, A2_batch = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_prop(Z1_batch, A1_batch, Z2_batch, A2_batch, W2, X_batch, one_hot_Y)

            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if (i % 10 == 0):
            Z1_train, A1_train, Z2_train, A2_train = forward_prop(W1, b1, W2, b2, X_train)
            accuracy_train = get_accuracy_batch(A2_train, y_train)
            print("iteration: ", i)
            print("accuracy: ", accuracy_train)

    return W1, b1, W2, b2

def main():
    n_max = 100
    X_train = read_images(TRAIN_DATA_FILENAME, n_max) 
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_max) 
    X_test = read_images(TEST_DATA_FILENAME, n_max)
    y_test = read_labels(TEST_LABELS_FILENAME, 1)

    X_train = np.array(X_train).T
    X_test = np.array(X_test).T
    
    # print(X_test.shape)
    # print(X_train.shape)
    W1, b1, W2, b2 = grad_descent(X_train, y_train, 100, 0.1, batch_size=10)
    
if __name__ == '__main__':
    main()