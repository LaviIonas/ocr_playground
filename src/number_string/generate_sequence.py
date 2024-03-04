from load_mnist_digits import MNIST 
from random import choice
import random
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
        # h, w = sequence.shape

        # canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # # Calculate the width of each segment based on the canvas size and number of digits
        # segment_width = 28 * w

        # for _ in range(max_segments):
        #     if w - h >= 0:  # Ensure space is available for the segment
        #         # Calculate random position for each segment within the canvas
        #         x_start = np.random.randint(0, canvas_w - h)
        #         x_end = x_start + h
        #         y_start = np.random.randint(0, canvas_h - h)
        #         y_end = y_start + h
                
        #        # Calculate the potential region of overlap
        #         overlap_region = canvas[y_start:y_end, x_start:x_end]

        #         # Check if there is any overlap between the segment and existing non-1 values
        #         if not np.any(overlap_region[sequence[:, :h] != 1]):
        #             # Overlay the segment onto the canvas, preserving original values
        #             canvas[y_start:y_end, x_start:x_end] = np.maximum(canvas[y_start:y_end, x_start:x_end], sequence[:, :h])

        #             # Overlay the segment onto the canvas, preserving original values
        #             canvas[y_start:y_end, x_start:x_end] = np.maximum(canvas[y_start:y_end, x_start:x_end], sequence[:, :h])

        #         # Remove the processed segment from the sequence
        #         sequence = sequence[:, h:]


        return canvas

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    m = MNIST_Sequence()
    sequence = [2,1,3]
    img_uniform = m.generate_image_sequence(sequence)
    canvas = m.generate_non_uniform_sequence(img_uniform)
    print(img_uniform)
    # print(img_uniform)
    show_image(canvas)

if __name__ == '__main__':
    main()