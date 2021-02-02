# %%
import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# %%
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 9, True)
        self.l2 = nn.Linear(9, 1, True)

    def forward(self, x):
        l1 = nn.functional.relu(self.l1(x))
        l2 = nn.functional.relu(self.l2(l1))
        return l2


# %%
class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 10, True),
                                   nn.Linear(10, 10, True),
                                   nn.Linear(10, 1, True)
                                   )

    def forward(self, x):
        res = torch.sigmoid(self.model(x))
        return res


# %%
gen = Gen()
dis = Dis()


# %%
def DLossFunc(xData, genData):
    out = torch.log(xData) + torch.log(1 - genData)
    return out.sum()


def GLossFunc(x):
    out = torch.log(1 - x).mean()
    return out.sum()


batchSize = 64
epochCount = 40000
lr = 0.001
prior_mu = -2.5
prior_std = 0.5
noise_range = 5.0


def sampleZ(batchSize, noizeRange):
    out = np.random.uniform(-noizeRange, noizeRange, size=[batchSize, 1])
    out = torch.Tensor(out)
    return out


def sampleX(batchSize, mu, std):
    out = np.random.normal(mu, std, size=[batchSize, 1])
    out = torch.Tensor(out)
    return out


criterion = nn.BCELoss()

optGen = torch.optim.Adam(gen.parameters(), lr=lr)
optDis = torch.optim.Adam(dis.parameters(), lr=lr)


def train(gen, dis, epochCount, batchSize, noizeRange, prior_mu, prior_std, noize_range):
    for epoch in range(epochCount):
        optDis.zero_grad()
        optGen.zero_grad()
        # discriminator train
        dis.zero_grad()
        # normal
        # real
        xBatch = sampleX(batchSize, prior_mu, prior_std)

        realDis = dis(xBatch)
        label = torch.full(realDis.size(), 1.0, dtype=torch.float)
        dLoss_real = criterion(realDis, label)
        dLoss_real.backward()

        # random
        # fake
        zBatch = sampleZ(batchSize, noizeRange=noizeRange)
        genXBatch = gen(zBatch)
        label.fill_(0.0)
        fakeDis = dis(genXBatch)
        dLoss_fake = criterion(fakeDis, label)
        dLoss_fake.backward()
        discriminatorLoss = dLoss_fake.item() + dLoss_real.item()
        optDis.step()

        gen.zero_grad()
        label.fill_(1.0)
        genXBatch = gen(zBatch)
        fakeDis = dis(genXBatch)
        gLoss = criterion(fakeDis, label)
        gLoss.backward()
        optGen.step()

        if epoch % (epochCount / 20) == 0:
            print('%d / %d Loss_D %.4f' % (epoch, epochCount, discriminatorLoss))



# train(gen, dis, int(1e5), batchSize, noise_range, prior_mu, prior_std, noise_range)
#
# x = sampleX(10, prior_mu, prior_std)
# y = gen(x).reshape(-1).detach()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(y.reshape(-1), ls='dashed', alpha=1, lw=3, color='b')
# ax.hist(x.reshape(-1), ls='dashed', alpha=0.5, lw=3, color='r')
# plt.show()
