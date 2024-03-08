def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def calculate_accuracy(y_test, y_pred):
    return (sum(
        [
            y_pred_i == y_test_i
            for y_pred_i, y_test_i in zip(y_test, y_pred)
        ]
    ) / len(y_test) ) * 100
