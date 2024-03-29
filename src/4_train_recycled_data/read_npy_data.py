import numpy as np
import os

DATA_DIR = "../../MNIST_RECYCLED"

def read_npy_files(data_dir):
    # Initialize empty dictionaries to store arrays
    data = {}

    # List all files in the directory
    files = os.listdir(data_dir)

    # Iterate over each file
    for f in files:
        # Check if the file is a .npy file
        if f.endswith('.npy'):
            # Load the array from the .npy file
            array_name = f.split('.')[0]
            array = np.load(os.path.join(data_dir, f))
            # Store the array in the dictionary
            data[array_name] = array

    return data

# def main():
    
#     data = read_npy_files(DATA_DIR)
#     print(data['X_train'].shape)
#     print(data['y_train'].shape)
#     print(data['X_test'].shape)
#     print(data['y_test'].shape)


# if __name__ == '__main__':
#     main()