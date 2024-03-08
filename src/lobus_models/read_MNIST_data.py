from ..0_opencv_playground.ocr_optimized import generate_data

def generate_MNIST_data(n_max, t_max):
    n = min(60000, n_max)
    t = min(10000, t_max)
    return generate_data(n, t)