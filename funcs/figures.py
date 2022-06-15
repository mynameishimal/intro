import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def makeCentralFig(test_img):
    plt.figure(figsize=(10,5))
    plt.imshow(test_img[1])
    plt.colormaps()
    plt.show()
    return

def makeNNdiagram():
    from nnv import NNV
    plt.rcParams["figure.figsize"] = 200,50
    layersList = [
    {"title":"Input\n(784 flatten)", "units": 784, "color": "Blue"},
    {"title":"Hidden 1\n(relu: 128)", "units": 128},
    {"title":"Output\n(softmax: 10)", "units": 10,"color": "Green"},
    ]   
    NNV(layersList, spacing_layer=10, max_num_nodes_visible=20, node_radius=1, font_size=24).render
    return

def printShirt(model):
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return
