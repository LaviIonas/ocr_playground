import cv2
import numpy as np
import matplotlib.pyplot as plt

from load_digits import get_digit_data

def define_roi(image_array):

    bounding_boxes = []

    for img in image_array:

        img_uint8 = np.uint8(img * 255)

        _, binary_img = cv2.threshold(img_uint8, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        bounding_boxes_per_image = []

        for c in contours:
            boundRect = cv2.boundingRect(c)
            x, y, w, h = boundRect
            area = w * h
            min_area = 50
            
            if area > min_area:
                bounding_boxes_per_image.append((x, y, w, h))
   
        bounding_boxes.append(bounding_boxes_per_image)

    return bounding_boxes

def resize_digits(image_array, bounding_boxes, margin=5):
    resized_digits = []

    for img, bbox_list in zip(image_array, bounding_boxes):
        img_uint8 = np.uint8(img * 255)

        for x, y, w, h in bbox_list:
            # Add margin to the bounding box
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin
            
            # Ensure that the modified bounding box is within the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_uint8.shape[1] - x)
            h = min(h, img_uint8.shape[0] - y)

            # Extract the digit using the modified bounding box coordinates
            digit = img_uint8[y:y+h, x:x+w]

            # Resize the digit to 28x28
            resized_digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

            # Invert the colors of the resized digit
            inverted_digit = cv2.bitwise_not(resized_digit)

            # Append the resized digit to the list of resized digits
            resized_digits.append(inverted_digit)

    return resized_digits

# def visualize_resized_digits(resized_digits, num_columns=5):
#     num_digits = len(resized_digits)
#     num_rows = (num_digits + num_columns - 1) // num_columns

#     plt.figure(figsize=(10, 2 * num_rows))

#     for i in range(num_digits):
#         plt.subplot(num_rows, num_columns, i + 1)
#         plt.imshow(resized_digits[i], cmap='gray')
#         plt.axis('off')

#     plt.show()

# def main():
#     X_train, y_train, X_test, y_test = get_digit_data()

    # ei = X_train[:3]

    # bounding_boxes = define_roi(ei)
    # resized_digits = resize_digits(ei, bounding_boxes, 3)

    # print(np.array(resized_digits).shape)

    # visualize_resized_digits(resized_digits)

# if __name__ == '__main__':
#     main()
