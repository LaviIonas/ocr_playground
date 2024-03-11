from MNIST_DATA import *

from random import choice
import random
import numpy as np
import idx2numpy
import os

class MNIST_SEQUENCE():
    def __init__(self, n, t):
        self.n = n
        self.dataset = generate_MNIST_data(self.n, t)
        self.images = self.dataset[0] 
        self.labels = self.dataset[1]
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

    def generate_non_uniform_sequence(self, seq):
        
        sequence, bounds = self.__generate_image_sequence(seq)

        h, w = 28, 28
        canvas_h = h + 10
        canvas_w = w + 10

        n_digits = (w//h)
        partition_width = canvas_w // n_digits

        digits_with_noise = []

        for digit, (l_bound, r_bound) in zip(sequence, bounds):
            canvas = np.ones((canvas_h, canvas_w), dtype=np.float32)

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

    def generate_MNIST_SEQ_data(self):
        dataset = []
        labels = []

        for i in range(self.n):
            sequence = np.random.randint(0,10, size=5)
            sequence_array = self.generate_non_uniform_sequence(seq=sequence)
            dataset.append(sequence_array)
            labels.append(sequence)

        return np.array(dataset), np.array(labels)