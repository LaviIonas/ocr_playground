import idx2numpy
import os

def get_digit_data():

    path = "../../MNIST_SEQUENCE/"

    X_train = idx2numpy.convert_from_file(os.path.join(path, "train-images.idx"))
    y_train = idx2numpy.convert_from_file(os.path.join(path, "train-labels.idx"))

    X_test = idx2numpy.convert_from_file(os.path.join(path, "test-images.idx"))
    y_test = idx2numpy.convert_from_file(os.path.join(path, "test-labels.idx"))

    return X_train, y_train, y_test, X_test

# def main():
#     X_train, y_train, y_test, X_test = get_digit_data()
#     print("X_train shape:", X_train.shape)
#     print("y_train shape:", y_train.shape)
#     print("X_test shape:", X_test.shape)
#     print("y_test shape:", y_test.shape)

# if __name__ == '__main__':
#     main()
