import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from classes.model import Model
from classes.network import Network


class BasicConvNet(Model):

    def __init__(self):
        super(BasicConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


if __name__ == "__main__":  # Required to allow multiprocessing on windows
    # loading the CIFAR-10 dataset
    # Please do not add the "data" folder to git.

    restore_model = True
    save_model = True
    save_root_path = os.path.join("saves", "cifar10")
    save_model_path = os.path.join(save_root_path, "model")

    if restore_model or save_model:
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

    model = BasicConvNet()
    loss_function = nn.CrossEntropyLoss
    optimizer = optim.Adam

    network = Network(model, loss_function, optimizer, 0.001, use_cuda=True)

    if restore_model:
        if os.path.isfile(save_model_path):
            network.load_params(save_model_path)

    # Train the model
    num_epochs = 5
    for i in range(num_epochs):
        network.training(trainloader, 1)
        network.validate(testloader)

    if save_model:
        network.save_params(save_model_path)
