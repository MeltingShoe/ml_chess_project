import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class SimpleCNN(nn.Module):
    def __init__(self, in_channels = 12, height = 8, width = 8):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.out = nn.Linear(256*8*8, 1)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.width, self.height)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output