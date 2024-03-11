from MNIST_DATA import generate_MNIST_data
from MNIST_SEQUENCE_DATA import MNIST_SEQUENCE
from digit_extractor import *

from knn_model import KNN_Model
from neural_model import NN

import numpy as np 
import matplotlib.pyplot as plt

def visualize_resized_digits(resized_digits, num_columns=5):
    num_digits = len(resized_digits)
    num_rows = (num_digits + num_columns - 1) // num_columns

    plt.figure(figsize=(10, 2 * num_rows))

    for i in range(num_digits):
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(resized_digits[i], cmap='gray')
        plt.axis('off')

    plt.show()

def main():
    '''
    Check MNIST Data Import
    '''
    # n = 10000
    # t = 1000
    # x1, y1, x2, y2 = generate_MNIST_data(n, t)
    # y1 = np.array(y1)
    # y2 = np.array(y2)

    '''
    Check MNIST SEQUENCE
    '''

    n1 = 11
    t1 = 100
    mnist_seq = MNIST_SEQUENCE(n1, t1)
    dataset, labels = mnist_seq.generate_MNIST_SEQ_data()

    # print(dataset.shape)
    # print(labels.shape)

    '''
    ROI
    '''

    x1,y1,x2,y2 = generate_dataset(3)
    x1 = np.array(x1)
    # x2 = np.array(x2)

    y1 = np.array(y1)
    # y2 = np.array(y2)

    print(x1.shape)
    print(y1.shape)

    '''
    knn
    '''

    # knn = KNN_Model("HOG", "KDTree", 5, n, t)
    # print(knn.check_accuracy(knn.knn_predict(x2), y2))

    '''
    NN
    '''

    # x1 = np.array(x1)
    # x2 = np.array(x2)

    # y1 = np.array(y1)
    # y2 = np.array(y2)

    # x1 = x1.reshape(x1.shape[0], -1)
    # x2 = x2.reshape(x2.shape[0], -1)

    # print(x1.shape)
    # print(y1.shape)

    # nn = NN(x1, y1, "CrossEntropy", [32,32], 10, 0.1, 200, 10, 0.9)
    # nn.initilize_layers()
    # model = nn.train()

if __name__ == '__main__':
    main()