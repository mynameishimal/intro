import tensorflow as tf
import matplotlib
import numpy
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def testModel(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Model Accuracy: {test_acc * 100}%")
    return
    
