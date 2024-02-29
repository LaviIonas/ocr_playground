import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

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

def knn(X_train, y_train, X_test, k=3):
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
    # X_train = extract_hog_features(X_train)
    # X_test = extract_hog_features(X_test)

    y_pred = knn(X_train, y_train, X_test, 5)
    # print(y_pred)
    # print(y_test)

    accuracy = (sum(
        [
            y_pred_i == y_test_i
            for y_pred_i, y_test_i in zip(y_test, y_pred)
        ]
    ) / len(y_test) ) * 100

    print(accuracy)

if __name__ == '__main__':
    main()