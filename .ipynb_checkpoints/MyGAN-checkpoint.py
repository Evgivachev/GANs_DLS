import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, numChannels):
        super(Generator,self).__init__()
        modules = []
        for i in range(len(numChannels) - 1):
            module = ResidualBlock(numChannels[i], numChannels[i])
            modules.append(module)
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
        self.activate = activation

    def forward(self, x):
        return x+self.conv(x)
