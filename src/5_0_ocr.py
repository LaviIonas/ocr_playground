#!/usr/bin/env pyhton

DATA_DIR = '../MNIST_DATA/'
TEST_DATA_FILENAME = DATA_DIR + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + '/train-images-idx3-ubyte/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + '/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        n_images = bytes_to_int(f.read(4))
        
        if n_max_images:
            n_images = n_max_images

        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_cols):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4) # magic number
        n_labels = bytes_to_int(f.read(4))
        
        if n_max_labels:
            n_labels = n_max_labels

        for labels_idx in range(n_labels):
            label = f.read(1)
            labels.append(label)
    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(x, y):
    # Euclidian distance

def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]
    

def knn(X_train, y_train, X_test, k=3):
    '''
    X_train => [img1, img2, img3, ...]
    y_train => [1,    3,    9,    ...]
    '''
    y_pred = []
    for sample in X_test:
        y_sample = ...
        y_pred.append(y_sample)
    return y_pred

def main():
    n_max = 100

    X_train = read_images(TRAIN_DATA_FILENAME, n_max)
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME, n_max)
    y_test = read_labels(TEST_LABELS_FILENAME)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)


if __name__ == '__main__':
    main()