from MNIST_SEQUENCE_DATA import generate_MNIST_SEQ_data

import cv2
import numpy as np

DATA_DIR = "../../MNIST_RECYCLED"

def __define_roi(image_array):

    bounding_boxes = []
    faulty_bbox = []

    for i, img in enumerate(image_array):

        img_uint8 = np.uint8(img * 255)

        _, binary_img = cv2.threshold(img_uint8, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        bounding_boxes_per_image = []

        for c in contours:
            boundRect = cv2.boundingRect(c)
            x, y, w, h = boundRect
            area = w * h
            min_area = 40
            
            if area > min_area:
                bounding_boxes_per_image.append((x, y, w, h))
   
        if len(bounding_boxes_per_image) == 5:
            bounding_boxes.append(bounding_boxes_per_image)
        else:
            faulty_bbox.append(i)

    return bounding_boxes, faulty_bbox

def resize_digits(image_array, labels, margin=5):
    bounding_boxes, faulty_bbox = __define_roi(image_array)

    for idx in faulty_bbox:
        image_array = np.delete(image_array, idx, axis=0)
        labels = np.delete(labels, idx, axis=0)

    resized_digits = []

    for img, bbox_list in zip(image_array, bounding_boxes):
        img_uint8 = np.uint8(img * 255)

        faulty_digits = []

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

    return np.array(resized_digits), np.array(labels)

def generate_dataset(MNIST_Sequence, m):
    X_train_arr, y_train_arr, X_test_arr, y_test_arr = generate_MNIST_SEQ_data()

    X_train, y_train = resize_digits(X_train_arr, y_train_arr, m)
    X_test, y_test = resize_digits(X_test_arr, y_test_arr, m)

    return X_train, y_train, X_test, y_test

def export_dataset():
    X_train, y_train, X_test, y_test = generate_dataset()

    # Save train images and labels
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)

    # Save test images and labels
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)

    print("Data Exported.")
