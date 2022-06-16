import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt 
from funcs import data, net, figures, callbacks
import argparse

data_name = "cifar10"
run_name = "cifar10"

def main_script(num_epochs=1,dataset_id="cifar10", use_cb = True, num_kernels=32, num_hidden = 64):
    (train_images, train_labels), (test_images, test_labels) = data.load_data(data_name = data_name)


    train_images, test_images = data.preprocess(train_images, test_images)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    model = net.buildModel(data_name=data_name, num_kernels=num_kernels, num_hidden=num_hidden)
    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    run_name = str(data_name) + "_" + str(num_epochs) + "epochs_" + str(num_kernels) + 'kernels_' + str(num_hidden) + 'hidden'

    history = model.fit(train_images, train_labels, epochs=num_epochs, callbacks = callbacks.tensorboard_cb(run_name=run_name),
                        validation_data=(test_images, test_labels), verbose=1)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)

    print(test_acc)

    # model = net.buildModel() 
    # net.fitModel(train_images, train_labels, num_epochs, model, use_cb, run_name)


    # test_perform_fname='test_perform_epo_'+str(num_epochs)
    # net.testModel(model, test_images, test_labels,fname=test_perform_fname)

    # net.makePredicts(model, test_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--num_epoch', type=int, help="number of epochs")
    parser.add_argument('-d','--dataset_id', help="dataset id")
    parser.add_argument('-k','--kern_num', type=int, help='number of kernels')
    parser.add_argument('-l', '--hiddenLayers', type=int, help='number hidden layers')
    args = parser.parse_args()
    main_script(args.num_epoch, args.dataset_id, True, args.kern_num, args.hiddenLayers)
# figures.makeCentralFig(test_images)

