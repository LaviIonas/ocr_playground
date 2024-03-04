from load_mnist_digits import MNIST 
from random import choice
import numpy as np
import matplotlib.pyplot as plt

class MNIST_Sequence():
    def __init__(self):
        self.dataset = MNIST()
        self.images, self.labels = self.dataset.load()
        self.label_map = {label: [] for label in range(10)}
        self.__generate_label_map()

    def __generate_label_map(self):
        for index, label in enumerate(self.labels):
            self.label_map[label].append(index)

    def __select_random_label(self, label):
        if self.label_map[label]:
            return choice(self.label_map[label])
        else:
            print(f"No images for the number {label} is available. \
                    Please try with a different number.")
            exit()

    def generate_image_sequence(self, sequence):
        images = [self.images[self.__select_random_label(label)] for label in sequence]
        image = np.hstack(images)
        return image

    def generate_non_uniform_sequence(self, sequence):
        h = sequence.shape[0]
        w = sequence.shape[1]

        canvas = np.zeros((h, w), dtype=np.uint8)
        offset = 0

        for i in range(0, int(w/h)):
            canvas[:, :28] = sequence[:, :28]

        return canvas

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    m = MNIST_Sequence()
    sequence = [2]
    img_uniform = m.generate_image_sequence(sequence)
    canvas = m.generate_non_uniform_sequence(img_uniform)
    print(canvas)
    # show_image(canvas)

if __name__ == '__main__':
    main()