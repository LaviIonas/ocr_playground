

def generate_dataset():
    pass

def main():
    X_train, y_train, y_test, X_test = get_digit_data()

    ei = X_train[:3]

    bounding_boxes = define_roi(ei)
    resized_digits = resize_digits(ei, bounding_boxes, 3)

    print(np.array(resized_digits).shape)

    visualize_resized_digits(resized_digits)

if __name__ == '__main__':
    main()
