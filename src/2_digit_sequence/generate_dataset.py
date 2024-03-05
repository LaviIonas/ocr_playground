from generate_sequence import MNIST_Sequence
import numpy as np 

def generate_database(n):
    dataset = []
    labels = []

    for i in range(n):
        sequence = np.random.randint(0,10, size=5)
        m = MNIST_Sequence()
        sequence_array = m.generate_non_uniform_sequence(seq=sequence)
        dataset.append(sequence_array)
        labels.append(sequence)

    return np.array(dataset), np.array(labels)
    

def main():
    train_dataset, train_labels = generate_database(n=5)
    print(train_dataset.shape)
    print(train_labels.shape)

if __name__ == '__main__':
    main()