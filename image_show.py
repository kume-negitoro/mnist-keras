import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

MODEL_INDEX = 1


def main():
    (x_train, y_train), _ = mnist.load_data()

    plt.title('label = ' + str(y_train[MODEL_INDEX]))
    plt.imshow(x_train[MODEL_INDEX])
    plt.show()


if __name__ == '__main__':
    main()