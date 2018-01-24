
from classes import utils
from classes.base_model import base_model
from models.feed_forward.example_cifar10_ff import BasicConvNet
from models.train.default_train import default_train

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


class runtime():

    def __init__(self, model):
        self.model = model

    def training_session(self, train_data, test_data, n_epochs):
        """function to manage the train session"""
        for epoch in range(self.start_epoch, n_epochs, 1):
            self.model.train(train_data, epoch)
            self.model.validate(test_data)

        self.save_params()

    def validate(self, dataloader):
        correct_count = 0
        total_count = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            inputs = Variable(inputs)

            outputs = self.model(inputs)
            max, argmax = torch.max(outputs.data, 1)
            predicted = argmax

            correct_count += (predicted == labels).sum()
            total_count += dataloader.batch_size

        print('Accuracy on the validation set: {0}'.format(
            100.0 * correct_count / total_count))


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
    training_method = default_train()
    evaluate = 0
    # loss_function = nn.CrossEntropyLoss
    # optimizer = optim.Adam

    network = base_model(feed_forward, training_method, evaluate,
                         use_cuda=True, resume=True, filepath=save_model_path)

    # Train the model
    num_epochs = 10
    network.training_session(trainloader, testloader, num_epochs)
