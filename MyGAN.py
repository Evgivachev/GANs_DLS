import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, numChannels):
        super(Generator, self).__init__()
        modules = []
        for i in range(len(numChannels) - 1):
            module = ResidualBlock(numChannels[i], numChannels[i+1])
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
        return x + self.activate(self.conv(x))


class Discriminator(nn.Module):
    def __init__(self, numChannels):
        super(Discriminator, self).__init__()
        modules = []
        for i in range(len(numChannels) - 1):
            module = ResidualBlock(numChannels[i], numChannels[i+1])
            modules.append(module)
        modules.append(nn.Sigmoid())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)
        return out


class GanModel(nn.Module):
    def __init__(self, gen, dis, criterion, device):
        super(GanModel, self).__init__()
        self.gen = gen.to(device)
        self.dis = dis.to(device)
        self.criterion = criterion
        self.device = device

    def forward(self, x):
        return self.gen(x)

    def sampleX(self, size, mu, std):
        out = np.random.normal(mu, std, size=size)
        out = torch.Tensor(out).to(self.device)
        return out

    def train(self, inputImages, targetImages, num_updaters=100, lr=3e-4):
        genOpt = torch.optim.Adam(self.gen.paramers(), lr=lr)
        disOpt = torch.optim.Adam(self.dis.paramers(), lr=lr)

        for updater in range(num_updaters):
            disOpt.zero_grad()
            realDis = self.dis(inputImages)
            label = torch.full_like(realDis, 1.0, dtype=torch.float)
            diss_Loss_real = self.criterion(realDis, label)
            diss_Loss_real.backward()

            # fake
            noise = self.sampleX(inputImages.size(), -2.5, 0.5)
            fakeDiss = self.dis(noise)
            torch.fill_(label, 0.0)
            diss_Loss_fake = self.criterion(fakeDiss, label)
            diss_Loss_fake.backward()

            discriminatorLoss = diss_Loss_real.item() + diss_Loss_fake.item()
            disOpt.step()

            #Generator
            genOpt.zero_grad()
            torch.fill_(1.0)
            fakeDis = self.dis(noise)
            generatorLoss = self.criterion(fakeDis,label)
            generatorLoss.backward()
            genOpt.step()

            if updater % (num_updaters / 20) == 0:
                print('%d / %d Loss_D %.4f' % (updater, num_updaters, discriminatorLoss))
                print('Loss_Generator %.4f' % (generatorLoss.ipem()))

