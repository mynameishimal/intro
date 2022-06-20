import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
from funcs import data, net, figures, callbacks
import argparse
import pandas as pd
import plotly.express as px


data_name = "cifar10"
run_name = "cifar10"

def main_script(num_epochs=1,dataset_id="cifar10", use_cb = True, num_kernels=32, num_hidden = 64):
    (train_images, train_labels), (test_images, test_labels) = data.load_data(data_name = data_name)

    train_images, test_images = data.preprocess(train_images, test_images)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    model = net.buildModel(data_name=data_name, num_kernels=num_kernels, num_hidden=num_hidden)

    weights = net.getLastLayerWeights(model)
    # pd.options.plotting.backend = "plotly"
    # df = pd.DataFrame(data=weights)
    # fig = df.plot()
    # fig.show()
    # print("hh")

    # print(np.shape(weights))
    # umap_3d = UMAP(n_components = 3)
    # proj_3d = umap_3d.fit_transform(features)
    X=weights[:3,:]
    df = pd.DataFrame(X, columns=class_names)
    fig_3d = px.scatter_3d(
        df, x=0, y=1, z=2
    )
    fig_3d.update_traces(marker_size=5)
    fig_3d.show()



    run_name = str(data_name) + "_" + str(num_epochs) + "epochs_" + str(num_kernels) + 'kernels_' + str(num_hidden) + 'hidden'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--num_epoch', type=int,default=30, help="number of epochs")
    parser.add_argument('-d','--dataset_id',default=0, help="dataset id")
    parser.add_argument('-k','--kern_num', type=int,default=32, help='number of kernels')
    parser.add_argument('-l', '--hiddenLayers', type=int, default=1, help='number hidden layers')
    args = parser.parse_args()
    main_script(args.num_epoch, args.dataset_id, True, args.kern_num, args.hiddenLayers)
