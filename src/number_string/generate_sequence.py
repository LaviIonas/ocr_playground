from load_mnist_digits import MNIST 

from random import choice
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

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

    def __determine_bounds(self, digit):
        left_bound = 0
        right_bound = 27

        # Find the left bound of the digit
        for i in range(28):
            if np.any(digit[:, i] != 1):
                left_bound = i - 1
                break

        # Find the right bound of the digit
        for i in range(27, -1, -1):
            if np.any(digit[:, i] != 1):
                right_bound = i + 1
                break

        return (left_bound, right_bound)

    def __generate_image_sequence(self, sequence):
        images = []
        bounds = []
        
        for label in sequence: 
            img = self.images[self.__select_random_label(label)] 
            bounds.append(self.__determine_bounds(img))
            images.append(img)

        return (images, bounds)

    def generate_non_uniform_sequence(self, seq, idx, precomputed_sequences):
        
        sequence, bounds = precomputed_sequences[idx]

        h, w = 28, 28
        canvas_h = h + 10
        canvas_w = w + 10

        n_digits = (w//h)
        partition_width = canvas_w // n_digits

        digits_with_noise = []

        for digit, (l_bound, r_bound) in zip(sequence, bounds):
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

            # random vertical pos
            y_pos = random.randint(0, 10)
            # random horizontal pos 
            x_pos = random.randint(-l_bound, canvas_w-r_bound)

            y_start = y_pos
            y_end = y_start + h

            # use v_pos and h_pos as starting position in canvas and translate digit
            if x_pos < 0:
                digit = np.array(digit[:, -x_pos:])
                canvas[y_start:y_end, 0:digit.shape[1]] = digit
            elif w+x_pos > canvas_w:
                offset = w + x_pos - canvas_w
                digit = digit[:, :w-offset]
                canvas[y_start:y_end, x_pos:canvas_w] = digit
            else:
                canvas[y_start:y_end, x_pos:x_pos+w] = digit

            # append canvas to array
            digits_with_noise.append(canvas)

        digits_with_noise = np.hstack(digits_with_noise)

        return digits_with_noise

    def generate_database(self, n):
        dataset = []
        labels = []

       # Precompute image sequences and bounds for each label
        precomputed_sequences = {}
        for label in range(10):
            precomputed_sequences[label] = [self.__generate_image_sequence([label]*5) for _ in range(n)]

        # Define a partial function with all required arguments
        partial_func = partial(self.generate_non_uniform_sequence, precomputed_sequences=precomputed_sequences)

        with Pool() as pool:
            results = pool.starmap(partial_func, [(self.generate_random_sequence(), idx, precomputed_sequences) for idx in range(n)])

        for result in results:
            dataset.append(result[0])
            labels.append(result[1])

        return np.array(dataset), np.array(labels)



def main():
    m = MNIST_Sequence()
    train_dataset, train_labels = m.generate_database(n=5)
    print(train_dataset.shape)
    print(train_labels.shape)

if __name__ == '__main__':
    main()
# def show_image(image):
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.show()