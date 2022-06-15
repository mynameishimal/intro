import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from funcs import data, net, figures

(train_images, train_labels), (test_images, test_labels) = data.load_fashion_mnist()

print(f"Train images dimensions: {train_images.shape}")
print(f"Test images dimensions: {test_images.shape}")

print('before',train_images[0][:])

train_images, test_images = data.preprocess_fashion_mnist(train_images, test_images)
print('after',train_images[0][:])
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=1)

net.testModel(model, test_images, test_labels)

predictions = model.predict(test_images)
predictions[1]

np.argmax(predictions[1])

test_labels[1]
figures.makeCentralFig(test_images)