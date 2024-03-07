from load_digits import get_digit_data
from extract_digits import define_roi, resize_digits

import os
import numpy as np

DATA_DIR = "../../MNIST_RECYCLED"

def generate_dataset():
    X_train_arr, y_train_arr, X_test_arr, y_test_arr = get_digit_data()

    bbox_train = define_roi(X_train_arr)
    bbox_test = define_roi(X_test_arr)

    margin_value = 3

    X_train = resize_digits(X_train_arr, bbox_train, margin_value)
    X_test = resize_digits(X_test_arr, bbox_test, margin_value)

    return X_train, y_train_arr, X_test, y_test_arr

def export_dataset():
    X_train, y_train, X_test, y_test = generate_dataset()

    # Save train images and labels
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)

    # Save test images and labels
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)

    print("Data Exported.")

def main():
    
    export_dataset()

if __name__ == '__main__':
    main()
