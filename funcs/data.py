import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def preprocess_fashion_mnist(train_images, test_images):
    train_images = train_images /255.
    test_images = test_images / 255
    return (train_images, test_images)

def print_dimensions(train_images, test_images):
    print(f"Train images dimensions: {train_images.shape}")
    print(f"Test images dimensions: {test_images.shape}")





#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
