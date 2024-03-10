from MNIST_data import generate_MNIST_data
from knn_model import KNN_Model
from neural_model import NN

import numpy as np 

def main():
    '''
    Check MNIST Data Import
    '''
    n = 10000
    t = 1000
    x1, y1, x2, y2 = generate_MNIST_data(n, t)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # knn = KNN_Model("HOG", "KDTree", 5, n, t)
    # print(knn.check_accuracy(knn.knn_predict(x2), y2))

    x1 = np.array(x1)
    x2 = np.array(x2)

    y1 = np.array(y1)
    y2 = np.array(y2)

    x1 = x1.reshape(x1.shape[0], -1)
    x2 = x2.reshape(x2.shape[0], -1)

    print(x1.shape)
    print(y1.shape)

    nn = NN(x1, y1, "CrossEntropy", [32,32], 10, 0.1, 200, 10, 0.9)
    nn.initilize_layers()
    model = nn.train()

if __name__ == '__main__':
    main()