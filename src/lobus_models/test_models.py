from MNIST_data import generate_MNIST_data
from knn_model import KNN_Model

import numpy as np 

def main():
    '''
    Check MNIST Data Import
    '''
    n = 1000
    t = 1000
    x1, y1, x2, y2 = generate_MNIST_data(n, t)

    # print(np.array(x1).shape)
    # print(np.array(y1).shape)
    # print(np.array(x2).shape)
    # print(np.array(y2).shape)

    knn = KNN_Model("LBP", "Basic", 5, n, t)
    print(knn.check_accuracy(knn.knn_predict(x2), y2))

if __name__ == '__main__':
    main()