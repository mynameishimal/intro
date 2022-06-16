from funcs import data, net, figures, callbacks
import sys
num_epochs=1
use_cb=True
dataset_id=0
run_name = "early_stop_tensorboard_fashion"
datasets=["fashion_mnist","digits_mnist"]
# (train_images, train_labels), (test_images, test_labels) = data.load_data(datasets[dataset_id])

def main_script(num_epochs=num_epochs,use_cb=use_cb,dataset_id=dataset_id):
    (train_images, train_labels), (test_images, test_labels) = data.load_data()


    train_images, test_images = data.preprocess_fashion_mnist(train_images, test_images)


    model = net.buildModel() 
    net.fitModel(train_images, train_labels, num_epochs, model, use_cb, run_name)


    test_perform_fname='test_perform_epo_'+str(num_epochs)
    net.testModel(model, test_images, test_labels,fname=test_perform_fname)

    net.makePredicts(model, test_images)


if __name__ == "__main__":
    # print(sys.argv[0]) 
    num_epochs=int(sys.argv[1])
    use_cb=sys.argv[2]
    dataset_id=int(sys.argv[3])
    main_script(num_epochs=num_epochs,use_cb=use_cb,dataset_id=dataset_id)
# figures.makeCentralFig(test_images)