from ast import Num
from funcs import data, net, figures, callbacks
import sys
import argparse
use_cb=True
dataset_id=0
data_name = "fashion_mnist"
run_name = "early_stop_tensorboard_fashion"
datasets=["fashion_mnist","digits_mnist"]
# (train_images, train_labels), (test_images, test_labels) = data.load_data(datasets[dataset_id])
parser = argparse.ArgumentParser(description='Process some integers.')

def main_script(num_epochs=1,use_cb=use_cb,dataset_id=dataset_id):
    print(num_epochs)
    # (train_images, train_labels), (test_images, test_labels) = data.load_data(data_name = data_name)


    # train_images, test_images = data.preprocess(train_images, test_images)


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
    args=parser.parse_args()
    print(args.num_epoch)


    # parser.add_argument('-use_cb', type=bool)
    # parser.add_argument('dataset_id', type = int)
    # parser.add_argument('num_kernels', type = int)
    # parser.add_argument('num_hidden', type = int)
    # args = parser.parse_args()
    # print(args.epoch_num)
    # main_script(num_epochs=num_epochs, use_cb=use_cb, dataset_id = dataset_id)
    
    
    # print(sys.argv[0]) 
    # num_epochs=int(sys.argv[1])
    # print(sys.argv[1])
    # use_cb=sys.argv[2]
    # dataset_id=int(sys.argv[3])
    # num_kernels = sys.argv[4]
    # num_hidden = sys.argv[5]
    # main_script(num_epochs=num_epochs,use_cb=use_cb,dataset_id=dataset_id)
    # figures.makeCentralFig(test_images)