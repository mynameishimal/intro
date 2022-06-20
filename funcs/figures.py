import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from funcs import net
import pandas as pd
from umap import UMAP
import plotly.express as px

def makeWeightsPlot(model, class_names):
    weights = net.getLastLayerWeights(model)
    pd.options.plotting.backend = "plotly"
    df = pd.DataFrame(data=weights)
    fig = df.plot()
    fig.show()
    print("hh")

    # print(np.shape(weights))
    # umap_3d = UMAP(n_components = 3)
    # proj_3d = umap_3d.fit_transform(features)
    X=weights[:,:3]
    fig_3d = px.scatter_3d(
        X, x=0, y=1, z=2
    )
    fig_3d.update_traces(marker_size=5)
    fig_3d.show()

def makeCentralFig(test_img):
    plt.figure(figsize=(10,5))
    plt.imshow(test_img[1])
    plt.colormaps()
    plt.show()
    return

def showData(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

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
