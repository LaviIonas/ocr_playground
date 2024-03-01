import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

from sklearn.neighbors import KDTree
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels

# directory shortcuts
DATA_DIR = '../MNIST_DATA/'
TEST_DIR = 'temp/'
TEST_DATA_FILENAME = DATA_DIR + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + '/train-images-idx3-ubyte/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

# convert bytes to ints
def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def calculate_accuracy(y_test, y_pred):
    return (sum(
        [
            y_pred_i == y_test_i
            for y_pred_i, y_test_i in zip(y_test, y_pred)
        ]
    ) / len(y_test) ) * 100

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
            images.append(image)
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

# Generate flat feature vector
def extract_flat_feature_vector(X):
    return [sample.flatten() for sample in X]

# LBP features
def extract_lbp_features(X):
    lbp_features = []
    for sample in X:
        # Compute LBP features
        lbp = local_binary_pattern(sample, P=8, R=1, method='uniform')

        # Flatten and Normalize Features
        flattened_features = lbp.flatten()
        normalized_features = (flattened_features - flattened_features.mean()) / flattened_features.std()
        
        lbp_features.append(normalized_features)
    return lbp_features

# HOG features
def extract_hog_features(X):
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

# Basic KNN
def basic_knn(X_train, y_train, X_test, k):
    y_pred = []

    for test_sample in X_test:
        # Caclulate distances between sample and all training samples
        distances = np.linalg.norm(X_train - test_sample, axis=1)

        # Get indices of k nearest neightbors
        nearest_indices = np.argsort(distances)[:k]

        # Get labels of k nearest neighbors
        nearest_labels = [y_train[idx] for idx in nearest_indices]

        # Find the most frequent labels
        pred_label = np.bincount(nearest_labels).argmax()

        y_pred.append(pred_label)

    return y_pred 

# Weighted KNN
def weighted_knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_sample in X_test:
        # Caclulate distances between sample and all training samples
        distances = np.linalg.norm(X_train - test_sample, axis=1)

        # Get indices of k nearest neightbors
        nearest_indices = np.argsort(distances)[:k]

        # Get labels of k nearest neighbors
        nearest_labels = [y_train[idx] for idx in nearest_indices]

        # Calculate weights based on inverse distances
        weights = 1 / distances[nearest_indices]

        # Weighted count of each class
        weighted_counts = np.bincount(nearest_labels, weights=weights, minlength=len(set(y_train)))

        # Predict the class with the heightest weight count
        pred_label = np.argmax(weighted_counts)

        y_pred.append(pred_label)

    return y_pred

# Radius Based KNN
def rad_knn(X_train, y_train, X_test, rad=0.5):
    y_pred = []
    for test_sample in X_test:
        # Caclulate distances between sample and all training samples
        distances = np.linalg.norm(X_train - test_sample, axis=1)

        # Find indices of samples within the specified radius
        within_rad_idx = np.where(distances <= rad)[0]

        if within_rad_idx.size == 0:
            # No samples around found, default prediction
            n_n_idx = np.argmin(distances)
            pred_label = y_train[n_n_idx]

        else:
            # Get labels of sampleswithin the radius 
            labels_within_rad = [y_train[idx] for idx in within_rad_idx]

            # Predict the most frequent label
            pred_label = max(set(labels_within_rad), key=labels_within_rad.count)

        y_pred.append(pred_label)

    return y_pred

# KDTree KNN
def kd_tree_knn(X_train, y_train, X_test, k):
    # Build KD-Tree on training data
    tree = KDTree(X_train)

    # Query KD-Tree for nearest neighbor
    _, indices = tree.query(X_test, k=k)

    y_pred = []
    for neighbor_idx in indices:
        # Get labels of nearest neighbors
        nearest_labels = [y_train[idx] for idx in neighbor_idx]

        # Make prediction based on majority vote
        pred_label = max(set(nearest_labels), key=nearest_labels.count)
        y_pred.append(pred_label)

    return y_pred

def main():
    n_max = 10000
    X_train = read_images(TRAIN_DATA_FILENAME, n_max) 
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_max) 
    X_test = read_images(TEST_DATA_FILENAME, 200)
    y_test = read_labels(TEST_LABELS_FILENAME, 200)

    # Flatten Feature Vector
    # X_train = extract_flat_feature_vector(X_train)
    # X_test = extract_flat_feature_vector(X_test)

    # LBP 
    # X_train = extract_lbp_features(X_train)
    # X_test = extract_lbp_features(X_test)

    # HOG
    X_train = extract_hog_features(X_train)
    X_test = extract_hog_features(X_test)

    # Basic KNN
    # y_pred = basic_knn(X_train, y_train, X_test, 3)

    # Weighted KNN
    # y_pred = weighted_knn(X_train, y_train, X_test, 3)

    # Radius-Based KNN
    # y_pred = rad_knn(X_train, y_train, X_test, 0.5)

    # KDTree KNN
    y_pred = kd_tree_knn(X_train, y_train, X_test, 3)

    print(calculate_accuracy(y_test, y_pred))

if __name__ == '__main__':
    main()