import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from funcs import callbacks, data

def testModel(model, test_images, test_labels, fname="model_accuracy"):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100}%")
    fname = fname + ".txt"
    numpy.savetxt(fname, (test_loss, test_acc))
    return

def buildModel():
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

    return model

def fitModel(train_images, train_labels, num_epochs, model, use_cb, run_name):
    if use_cb:
        history = model.fit(train_images, train_labels, epochs=num_epochs, callbacks = [callbacks.early_cb(), callbacks.tensorboard_cb(run_name)], validation_split=0.2, verbose =1)
    else:
        history = model.fit(train_images, train_labels, epochs=num_epochs, validation_split=0.2, verbose =1)
    return

def makePredicts(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions[1])

    
