from ..0_opencv_playground.ocr_optimized import *

def train_knn_model(feature_func, knn_func, k, n_max, t_max):

    X_train,  y_train, X_test, y_test = generate_data(n_max, t_max)

    X_train = feature_func(X_train)
    X_test = feature_func(X_test)

    y_pred = knn_func(X_train, y_train, X_test, k)

    accuracy = calculate_accuracy(y_test, y_pred)

    print("feature: {feature_func} at knn: {knn_func} a: {accuracy}")