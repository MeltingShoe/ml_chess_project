from classes.base_model import generate_class
from models.evaluate.supervised_evaluate import supervised_evaluate
from models.feed_forward.example_cifar10_ff import BasicConvNet
from models.train.default_train import default_train

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class runtime():

    def __init__(self, model):
        self.model = model

    def training_session(self, train_data, test_data, n_epochs):
        self.model.training_session(train_data, test_data, n_epochs)


if __name__ == "__main__":  # Required to allow multiprocessing on windows
    # loading the CIFAR-10 dataset
    # Please do not add the "data" folder to git.

    restore_model = True
    save_root_path = os.path.join("saves", "cifar10")
    save_model_path = os.path.join(save_root_path, "model")

    if restore_model:
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)

    data_path = os.path.join("data", "cifar10")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=96,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=96,
                                             shuffle=False, num_workers=4)

    feed_forward = BasicConvNet()
    training_method = default_train
    evaluate = supervised_evaluate
    base_model = generate_class(feed_forward, training_method, supervised_evaluate)
    network = base_model(feed_forward)

    # Train the model
    num_epochs = 10
    runtime_obj = runtime(network)
    runtime_obj.training_session(trainloader, testloader, num_epochs)
