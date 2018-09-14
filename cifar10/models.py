import torch
import torch.nn as nn
from torch import nn
import torch.nn.modules.normalization as tn
from torch.distributed import group


class AlexNetCifa10(nn.Module):
    def __init__(self):
        """
        :param layer_dim: list of dims of all layers
        :param activation: an activation function (nn.Sigmoid(), nn.ReLU())
        """
        super(AlexNetCifa10, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, padding=2, groups=1)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, groups=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)

        self.LRN = tn.CrossMapLRN2d(5)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(256 * 8 * 8, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.last = nn.Linear(1024, 10)

    def forward(self, x):
        # print(x.shape)
        b = x.shape[0]
        c = 3
        h, w = 32, 32

        x = x.view(b, c, h, w)  # input  3x32x32
        # print(x.shape)
        x = self.LRN(self.activation(self.conv1(x)))  # ..... 48x32x32

        # print(x.shape)
        x = self.pooling(self.LRN(self.activation(self.conv2(x))))  # ..... 128x16x16
        # print(x.shape)
        x = self.activation(self.conv3(x))  # ..... 192x16x16
        # print(x.shape)
        x = self.activation(self.conv4(x))  # ..... 192x16x16
        # print(x.shape)
        x = self.activation(self.conv5(x))  # ..... 128x16x16
        # print(x.shape)
        x = self.pooling(x)  # ..... 128x8x8
        # print(x.shape)

        x = x.view(b, -1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        result = self.last(x)
        return result


class AlexNetCifa10_2(nn.Module):
    def __init__(self):
        """
        :param layer_dim: list of dims of all layers
        :param activation: an activation function (nn.Sigmoid(), nn.ReLU())
        """
        super(AlexNetCifa10_2, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, padding=2, groups=1)
        self.conv2 = nn.Conv2d(96, 384, kernel_size=5, padding=2, groups=2)
        self.conv3 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)

        self.LRN = tn.CrossMapLRN2d(5)
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(256 * 8 * 8, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.last = nn.Linear(1024, 10)

    def forward(self, x):
        # print(x.shape)
        b = x.shape[0]
        c = 3
        h, w = 32, 32

        x = x.view(b, c, h, w)      # input  96x32x32
        # print(x.shape)
        l1 = self.conv1(x)          # ..... 384x32x32
        x = self.activation(l1)

        l2 = self.conv2(x)          # ..... 384x16x16
        x = self.pooling(self.activation(l2))

        l3 = self.conv3(x)          # ..... 384x16x16
        x = self.activation(l3)

        l4 = self.conv4(x) + l2     # ..... 384x16x16
        x = self.activation(l4)

        l5 = self.conv5(x)          # ..... 256x16x16
        x = self.pooling( self.activation(l5) )

        x = x.view(b, -1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        result = self.last(x)
        return result