import tensorflow as tf
import matplotlib
import numpy
import datetime
import nnv

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def tensorboard_cb(run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    log_dir = "logs/fit/" + run_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback
    
def early_cb(monitor='val_accuracy', patience=6):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)

