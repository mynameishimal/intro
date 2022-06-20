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

def buildFashionModel():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def buildCifarCNN(num_kernels, num_hidden,intLayerName='secondLast'):
    model = keras.models.Sequential()
    layers = keras.layers
    model.add(layers.Conv2D(num_kernels, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(num_kernels*2, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(num_kernels*2, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_hidden, activation='relu',name=intLayerName))
    model.add(layers.Dense(10))
    return model

def buildModel(data_name = "fashion_mnist", num_kernels=32, num_hidden=64,intLayerName='secondLast'):
    switcher = {
        "fashion_mnist": 0,
        "mnist_digits": 1,
        "cifar10": 2,
        "cifar100": 3,
    }

    arg= switcher.get(data_name)
    match arg:
        case 0:
            return buildFashionModel();
        case 1:
            return;
        case 2:
            return buildCifarCNN(num_kernels, num_hidden,intLayerName);
        case 3:
            return;

def fitModel(train_images, train_labels, num_epochs, model, use_cb, run_name):
    if use_cb:
        history = model.fit(train_images, train_labels, epochs=num_epochs, callbacks = [callbacks.early_cb(), callbacks.tensorboard_cb(run_name)], validation_split=0.2, verbose =1)
    else:
        history = model.fit(train_images, train_labels, epochs=num_epochs, validation_split=0.2, verbose =1)
    return

def makePredicts(model, test_images):
    predictions = model.predict(test_images)
    return np.argmax(predictions[1])

def getLastLayerWeights(model):
    num_layers = len(model.layers)
    return model.layers[num_layers-1].get_weights()[0]

    
def getInterOutModel(model,layerName):
    return keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layerName).output)


    
