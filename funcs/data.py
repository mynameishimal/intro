import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def load_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def preprocess_fashion_mnist(train_images, test_images):
    train_images = train_images /255.
    test_images = test_images / 255
    return (train_images, test_images)



#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
