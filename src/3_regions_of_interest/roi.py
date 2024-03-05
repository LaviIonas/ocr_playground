import cv2
import numpy as np

from load_digits import get_digit_data

def define_roi(image_array):
    for idx, img in enumerate(image_array):

        img_uint8 = np.uint8(img * 255)

        thresh_val, binary_img = cv2.threshold(img_uint8, 200, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for _, c in enumerate(contours):

            boundRect = cv2.boundingRect(c)


    #     for i in range(1, num_labels):
    #         x, y, w, h, area = stats[i]

    #         if area > 0:
    #             cv2.rectangle(img_uint8, (x,y), (x+w, y+h), (0, 255, 0), 2)
    #             roi = binary_img[y:y+h, x:x+w]

    #     cv2.imshow("Result", img_uint8)
    #     cv2.waitKey(1000)
    # cv2.destroyAllWindows()

def main():
    X_train, y_train, y_test, X_test = get_digit_data()
    # print("X_train shape:", X_train[:1].shape)
    define_roi(X_train[:1])


if __name__ == '__main__':
    main()
