import struct
from array import array
import numpy as np

DATA_DIR = '../../MNIST_DATA'
TEST_DATA_FILENAME = DATA_DIR + '/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + '/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

class MNIST():
    def __init__(self):
        self.name_img = TEST_DATA_FILENAME
        self.name_lbl = TEST_LABELS_FILENAME

    def load(self):
        labels = []
        images = []

        with open(self.name_lbl, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))

            labels = array("B", f.read())

        with open(self.name_img, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

            image_data = list(map(lambda pixel: (255 - pixel) / 255.0, array("B", f.read())))

            images = np.asarray(image_data, dtype=np.float32).reshape(size, rows, cols)

        return images, labels



# def main():
#     m = MNIST()
#     images, labels = m.load()
#     print(images[0])

# if __name__ == '__main__':
#     main()