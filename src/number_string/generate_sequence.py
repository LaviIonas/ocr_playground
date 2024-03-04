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
        if len(self.label_map[label]) > 0:
            return choice(self.label_map[label])
        else:
            print("No images for the number " + str(label) +
                  " is available. Please try with a different number.")
            exit()

    def generate_image_sequence(self, sequence):
        sequence_length = len(sequence)
        image_sequence = []
        for digit in sequence:
            random_label_number = self.__select_random_label(digit)
            image_sequence.append(self.images[random_label_number])
            return np.hstack(image_sequence)

    def generate_sequence_with_noise(self, sequence, max_shift=2):
        sequence_length = len(sequence)
        image_sequence = []
        for digit in sequence:
            random_label_number = self.__select_random_label(digit)
            noisy_image = self.add_random_noise(self.images[random_label_number], max_shift)
            image_sequence.append(noisy_image)
        return np.hstack(image_sequence)

    def add_random_noise(self, image, max_shift):
        noisy_image = image.copy()
        shift = np.random.randint(-max_shift, max_shift+1)
        noisy_image = np.roll(noisy_image, shift, axis=0)
        shift = np.random.randint(-max_shift, max_shift+1)
        noisy_image = np.roll(noisy_image, shift, axis=1)
        return noisy_image

def print_sequence(sequence):
    fig, axes = plt.subplots(1, len(sequence), figsize=(len(sequence)*2, 2))
    for i, digit in enumerate(sequence):
        axes[i].imshow(digit, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Digit {i+1}')
    plt.show() 

def main():
    m = MNIST_Sequence()
    sequence = [0]  # Example sequence
    sequence_with_noise = m.generate_sequence_with_noise(sequence)
    sequence_without_noise = m.generate_image_sequence(sequence)
    print_sequence(sequence_with_noise)
    print_sequence(sequence_without_noise)

if __name__ == '__main__':
    main()