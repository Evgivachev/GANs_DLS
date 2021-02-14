import torch
from torch import nn
from torchvision import transforms, datasets
from MyGAN import *
from ResnetGenerator import *
from NLayerDiscriminator import *
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from MyDataset import *
from pathlib import Path
from torchsummary import summary

device = torch.device("cuda")
size = 64
batchSize = 1
learningRate = 2e-4
LAMBDA = 10
horseFolder = Path(r'D:\Datasets\horse2zebra\trainA')
zebraFolder = Path(r'D:\Datasets\horse2zebra\trainB')

horseDataset = MyDataset(horseFolder, size)
zebraDataset = MyDataset(zebraFolder, size)
transform = transforms.Compose(
    [transforms.Resize(size),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])
horseLoader = torch.utils.data.DataLoader(horseDataset, batch_size=batchSize, shuffle=True)
zebraLoader = torch.utils.data.DataLoader(zebraDataset, batch_size=batchSize, shuffle=True)
num_blocks = 6
generator_A = ResnetGenerator(num_blocks=num_blocks).to(device)
discriminator_A = NLayerDiscriminator(input_nc=3).to(device)
generator_A_Opt = torch.optim.Adam(generator_A.parameters(), lr=learningRate)
discriminator_A_Opt = torch.optim.Adam(discriminator_A.parameters(), lr=learningRate)

generator_B = ResnetGenerator(num_blocks=num_blocks).to(device)
discriminator_B = NLayerDiscriminator(input_nc=3).to(device)
generator_B_Opt = torch.optim.Adam(generator_B.parameters(), lr=learningRate)
discriminator_B_Opt = torch.optim.Adam(discriminator_B.parameters(), lr=learningRate)


def generatorBSE_Loss(image):
    ones_label = torch.ones_like(image, dtype=torch.float, device=device)
    criterionBCE = torch.nn.BCELoss()
    return criterionBCE(image, ones_label)


def discriminatorBCE_Loss(real, fake):
    ones = torch.ones_like(real, dtype=torch.float, device=device)
    zeros = torch.zeros_like(fake, dtype=torch.float, device=device)
    criterionBCE = torch.nn.BCELoss()
    return criterionBCE(real, ones) + criterionBCE(fake, zeros)


def calc_cycle_loss(real_image, cycled_image):
    loss = torch.nn.L1Loss()
    return loss(real_image, cycled_image) * LAMBDA


def identity_loss(real_image, same_image):
    loss = torch.nn.L1Loss()
    return LAMBDA * 0.5 * loss(real_image, same_image)


def train(num_epoch):
    discriminatorEpochLosses = []
    generatorEpochLosses = []
    for epoch in range(num_epoch):
        print("epoch ", epoch)
        discriminatorBatchLosses = []
        generatorBatchLosses = []
        for horseBatch, zebraBatch in zip(horseLoader, zebraLoader):
            
            horseBatch = horseBatch.to(device)
            zebraBatch = zebraBatch.to(device)

            zebraFake = generator_A(horseBatch)
            horseFake = generator_B(zebraBatch)

            horseRestored = generator_B(zebraFake)
            zebraRestored = generator_A(horseFake)

            sameHorse = generator_B(horseBatch)
            sameZebra = generator_A(zebraBatch)

            isZebra = discriminator_A(zebraFake)
            isHorse = discriminator_B(horseFake)

            generator_A_Loss = generatorBSE_Loss(isZebra)
            generator_B_Loss = generatorBSE_Loss(isHorse)
            
            generator_A_identifyLoss = identity_loss(zebraBatch, sameZebra)
            generator_B_identityLoss = identity_loss(horseBatch, sameHorse)

            totalCycleLoss = calc_cycle_loss(zebraBatch, zebraRestored) + calc_cycle_loss(horseBatch, horseRestored)

            generatorsLoss = generator_A_Loss + generator_B_Loss + totalCycleLoss + generator_A_identifyLoss+ generator_B_identityLoss
            generatorBatchLosses.append(generatorsLoss.item())

            generator_A.zero_grad()
            generator_B.zero_grad()
            generatorsLoss.backward()
            generator_A_Opt.step()
            generator_B_Opt.step()
            
            

            zebraFake = generator_A(horseBatch)
            horseFake = generator_B(zebraBatch)

            isZebra_false = discriminator_A(zebraFake)
            isHorse_false = discriminator_B(horseFake)

            isZebra_true = discriminator_A(zebraBatch)
            isHorse_true = discriminator_B(horseBatch)

            discriminator_A_Loss = discriminatorBCE_Loss(isZebra_true, isZebra_false)
            discriminator_B_Loss = discriminatorBCE_Loss(isHorse_true, isHorse_false)

            discriminatorBatchLosses.append(discriminator_A_Loss.item() +
                                            discriminator_B_Loss.item())

            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            discriminator_A_Loss.backward()
            discriminator_B_Loss.backward()
            discriminator_A_Opt.step()
            discriminator_B_Opt.step()

        generatorEpochLosses.append(sum(generatorBatchLosses) / len(generatorBatchLosses))
        discriminatorEpochLosses.append(sum(discriminatorBatchLosses) / len(discriminatorBatchLosses))

        plt.subplot(121)

        plt.plot(generatorEpochLosses)
        plt.title = "GeneratorLoss"
        plt.subplot(122)
        plt.plot(discriminatorEpochLosses)
        plt.title = "DiscriminatorLoss"
        plt.show()

        plt.subplot(131)
        image = transformTensorToImage(horseBatch[0])
        plt.title = 'Input'
        plt.imshow(image)

        plt.subplot(132)
        image = transformTensorToImage(zebraFake[0])
        plt.title = 'Transformed'
        plt.imshow(image)

        plt.subplot(133)
        image = transformTensorToImage(horseRestored[0])
        plt.title = 'Restored'
        plt.imshow(image)
        plt.show()

        plt.subplot(131)
        image = transformTensorToImage(zebraBatch[0])
        plt.title = 'Input'
        plt.imshow(image)

        plt.subplot(132)
        image = transformTensorToImage(horseFake[0])
        plt.title = 'Transformed'
        plt.imshow(image)

        plt.subplot(133)
        image = transformTensorToImage(zebraRestored[0])
        plt.title = 'Restored'
        plt.imshow(image)
        plt.show()

        # imshow(horseBatch[0].detach().cpu(), plt_ax=plt.subplot(131))
        # imshow(zebraFake[0].detach().cpu(), plt_ax=plt.subplot(122))

        torch.save(generator_A, '_generator_A_epoch%s' % epoch)
        torch.save(generator_B, '_generator_B_epoch%s' % epoch)
        torch.save(discriminator_A, '_discriminator_A_epoch%s' % epoch)
        torch.save(discriminator_B, '_discriminator_B_epoch%s' % epoch)

        # # zebra to discriminator
        # discriminator_A_Opt.zero_grad()
        # realDis = discriminator_A(zebraBatch)
        # label = torch.full_like(realDis, 1.0, dtype=torch.float).to(device)
        # realDisLoss = criterion(realDis, label)


#
# # horse to generator
# fakeZebra = generator_A(horseBatch)
# fakeDis = discriminator_A(fakeZebra)
# label = torch.full_like(fakeDis, 0.0, dtype=torch.float).to(device)
# fakeDisLoss = criterion(fakeDis, label)
# generalDiscriminatorLoss = realDisLoss + fakeDisLoss
# generalDiscriminatorLoss.backward()
# discriminator_A_Opt.step()
#
# generator_A_Opt.zero_grad()
# fakeZebra = generator_A(horseBatch)
# fakeDis = discriminator_A(fakeZebra)
# generatorLoss = criterion(fakeDis, torch.full_like(fakeDis, 1, dtype=torch.float).to(device))
# generatorLoss.backward()
# generator_A_Opt.step()


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
    plt.show()


def transformTensorToImage(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.detach().cpu()
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


# summary(generator_A, (3, 128, 128), batch_size=4)
# for horseBatch, zebraBatch in zip(horseLoader, zebraLoader):
#     imshow(horseBatch[0].detach(), plt_ax=plt.subplot(121))
#     imshow(horseBatch[1].detach(), plt_ax=plt.subplot(122))
#     break

train(40)
