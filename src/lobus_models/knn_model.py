from MNIST_DATA import generate_MNIST_data

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

from sklearn.neighbors import KDTree

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
        self.X_train = self.get_features(self.X_train)

        self.knn_func = knn_func
        self.sup_knn = ["Basic", "Weighted", "Radius", "KDTree"]
        self.check_supported_knn()

    def check_supported_feature(self):
        if self.feature_extraction not in self.sup_features:
            raise ValueError("Unsupported feature extraction method. Supported methods are: {}".format(self.sup_features))

    def check_supported_knn(self):
        if self.knn_func not in self.sup_knn:
            raise ValueError("Unsupported knn method. Supported methods are: {}".format(self.sup_knn))

    def get_features(self, X):
        features = []

        if self.feature_extraction == "HOG":
            features = self.hog(X)
        elif self.feature_extraction == "LBP":
            features = self.lbp(X)

        return features

    def hog(self, X):
        hog_features = []
        for sample in X:
            # Compute HOG features
            hog_feature = hog(sample, orientations=9, 
                            pixels_per_cell=(8,8), 
                            cells_per_block=(2,2),
                            block_norm='L2-Hys')
            
            # Flatten Features
            flattened_features = hog_feature.flatten()
            hog_features.append(flattened_features)

        return hog_features

    def lbp(self, X):
        lbp_features = []
        for sample in X:
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
            y_pred = self.weighted_knn(X_test)
        elif self.knn_func == "Radius":
            y_pred = self.rad_knn(X_test)
        elif self.knn_func == "KDTree":
            y_pred = self.kd_tree_knn(X_test)

        return y_pred

    def basic_knn(self, X_test):

        X_test = self.get_features(X_test)

        y_pred = []

        for test_sample in X_test:
            # Caclulate distances between sample and all training samples
            distances = np.linalg.norm(self.X_train - test_sample, axis=1)

            # Get indices of k nearest neightbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Get labels of k nearest neighbors
            
            nearest_labels = [self.y_train[idx] for idx in nearest_indices]

            # Find the most frequent labels
            pred_label = np.bincount(nearest_labels).argmax()
            y_pred.append(pred_label)

        return y_pred 

    def weighted_knn(self, X_test):

        X_test = self.get_features(X_test)

        y_pred = []
        
        for test_sample in X_test:
            # Caclulate distances between sample and all training samples
            distances = np.linalg.norm(self.X_train - test_sample, axis=1)

            # Get indices of k nearest neightbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Get labels of k nearest neighbors
            nearest_labels = [self.y_train[idx] for idx in nearest_indices]

            # Calculate weights based on inverse distances
            weights = 1 / distances[nearest_indices]

            # Weighted count of each class
            weighted_counts = np.bincount(nearest_labels, weights=weights, minlength=len(set(self.y_train)))

            # Predict the class with the heightest weight count
            pred_label = np.argmax(weighted_counts)

            y_pred.append(pred_label)

        return y_pred

    def rad_knn(self, X_test):

        X_test = self.get_features(X_test)

        y_pred = []
        
        for test_sample in X_test:
            # Caclulate distances between sample and all training samples
            distances = np.linalg.norm(self.X_train - test_sample, axis=1)

            # Find indices of samples within the specified radius
            within_rad_idx = np.where(distances <= self.k)[0]

            if within_rad_idx.size == 0:
                # No samples around found, default prediction
                n_n_idx = np.argmin(distances)
                pred_label = self.y_train[n_n_idx]

            else:
                # Get labels of sampleswithin the radius 
                labels_within_rad = [self.y_train[idx] for idx in within_rad_idx]

                # Predict the most frequent label
                pred_label = max(set(labels_within_rad), key=labels_within_rad.count)

            y_pred.append(pred_label)

        return y_pred

    def kd_tree_knn(self, X_test):

        X_test = self.get_features(X_test)

        # Build KD-Tree on training data
        tree = KDTree(self.X_train)

        # Query KD-Tree for nearest neighbor
        _, indices = tree.query(X_test, k=self.k)

        y_pred = []

        for neighbor_idx in indices:
            # Get labels of nearest neighbors
            nearest_labels = [self.y_train[idx] for idx in neighbor_idx]

            # Make prediction based on majority vote
            pred_label = max(set(nearest_labels), key=nearest_labels.count)
            y_pred.append(pred_label)

        return y_pred

    def check_accuracy(self, y_test, y_pred):
        return (sum(
            [
                y_pred_i == y_test_i
                for y_pred_i, y_test_i in zip(y_test, y_pred)
            ]
        ) / len(y_test) ) * 100