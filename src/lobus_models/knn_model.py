from MNIST_data import generate_MNIST_data

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

class KNN_Model():
    def __init__(self, feature_extraction, knn_func, k, n, t):
        self.k = k
        
        # Init Data
        self.data = generate_MNIST_data(n, t)
        self.X_train = self.data[0]
        self.y_train = self.data[1]

        # Init Features
        self.feature_extraction = feature_extraction
        self.sup_features = ["HOG", "LBP"] 
        self.check_supported_feature()
        self.features = self.get_features()

        self.knn_func = knn_func
        self.sup_knn = ["Basic", "Weighted", "Radius", "KDTree"]
        self.check_supported_knn()

    def check_supported_feature(self):
        if self.feature_extraction not in self.sup_features:
            raise ValueError("Unsupported feature extraction method. Supported methods are: {}".format(self.sup_features))

    def check_supported_knn(self):
        if self.knn_func not in self.sup_knn:
            raise ValueError("Unsupported knn method. Supported methods are: {}".format(self.sup_knn))

    def get_features(self):
        features = []

        if self.feature_extraction == "HOG":
            features = self.hog()
        elif self.feature_extraction == "LBP":
            features = self.lbp()

        return features

    def hog(self):
        hog_features = []
        for sample in self.X_train:
            # Compute HOG features
            hog_feature = hog(sample, orientations=9, 
                            pixels_per_cell=(8,8), 
                            cells_per_block=(2,2),
                            block_norm='L2-Hys')
            
            # Flatten Features
            flattened_features = hog_feature.flatten()
            hog_features.append(flattened_features)

        return hog_features

    def lbp(self):
        lbp_features = []
        for sample in self.X_train:
            # Compute LBP features
            lbp = local_binary_pattern(sample, P=8, R=1, method='uniform')

            # Flatten and Normalize Features
            flattened_features = lbp.flatten()
            normalized_features = (flattened_features - flattened_features.mean()) / flattened_features.std()
            
            lbp_features.append(normalized_features)
        return lbp_features

    def knn_predict(self, X_test):
        y_pred = []

        if self.knn_func == "Basic":
            y_pred = self.basic_knn(X_test)
        elif self.knn_func == "Weighted":
            pass
        elif self.knn_func == "Radius":
            pass
        elif self.knn_func == "KDTree":
            pass

        return y_pred

    def basic_knn(self, X_test):
        y_pred = []

        for test_sample in X_test:
            # Caclulate distances between sample and all training samples
            distances = np.linalg.norm(self.X_train - test_sample, axis=1)

            # Get indices of k nearest neightbors
            nearest_indices = np.argsort(distances)[:self.k]
            print(nearest_indices)
            # Get labels of k nearest neighbors
            nearest_labels = [self.y_train[idx] for idx in nearest_indices]

            # Find the most frequent labels
            pred_label = np.bincount(nearest_labels).argmax()

            y_pred.append(pred_label)

        return y_pred 

    def check_accuracy(self, y_test, y_pred):
        return (sum(
            [
                y_pred_i == y_test_i
                for y_pred_i, y_test_i in zip(y_test, y_pred)
            ]
        ) / len(y_test) ) * 100