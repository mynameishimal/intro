import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def load_data(data_name = "fashion_mnist"):

    switcher = {
        "fashion_mnist": 0,
        "mnist_digits": 1,
        "cifar10": 2,
        "cifar100": 3,
    }
    arg= switcher.get(data_name)
    match arg:
        case 0:
            return tf.keras.datasets.fashion_mnist.load_data()
        case 1:
            return tf.keras.datasets.mnist.load_data(path="mnist.npz")
        case 2:
           
            # test=tf.keras.datasets.cifar10.load_data()
            # print(test)
            return tf.keras.datasets.cifar10.load_data()
        case 3:
            return tf.keras.datasets.cifar100.load_data(label_mode="fine")



def preprocess(train_images, test_images):
    train_images = train_images /255.
    test_images = test_images / 255
    return (train_images, test_images)

def print_dimensions(train_images, test_images):
    print(f"Train images dimensions: {train_images.shape}")
    print(f"Test images dimensions: {test_images.shape}")





#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
