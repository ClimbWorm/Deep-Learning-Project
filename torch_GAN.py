import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_G import G
from torch_D import D
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# for discriminator:
# loss function from noise: E = -ln(1-D(x)) want 0
# loss function from real images: E = -ln(D(x)) want 1
# for generator:
# loss function from noise: E = -ln(D(G(z))) want 1 (want discriminator unable to do it right)

LR = 0.0002
BETA1 = 0.5  # decay of first order momentum of gradient
BETA2 = 0.999  # decay of second order momentum of gradient
EPOCHS = 200
LATENT_DIM = 100
SAVE = 100  # save generated image every 100 images

transform = Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

dataset_train = datasets.ImageFolder(root='./unit_test_sample/', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=64,
                                           shuffle=True)

# class GAN(nn.Module):
#     def __init__(self,g_model,d_model):
#         super(GAN,self).__init__()
#         self.G = g_model
#         self.D = d_model
#
#     def forward(self,x):
#
#         out = self.G(x)
#         out = self.D(x)
#         return out
#
#
#
#     def train(self):
#         optimizer = optim.RMSprop(lr=0.0002,momentum=0.5)
#         loss = nn.BCELoss()
#         with self.D.no_grad():


if __name__ == "__main__":

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = G()
    discriminator = D()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    discriminator_loss = []
    generator_loss = []
    for epoch in range(EPOCHS):
        for i, (imgs, _) in enumerate(train_loader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # print("image size: ",imgs.size(0))

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (LATENT_DIM))))  # batch size = imgs.shape[0]
            # print(z)

            # Generate a batch of images
            gen_imgs = generator(z)
            # print('discriminator',discriminator(gen_imgs).shape)
            # print('valid',valid.shape)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            discriminator_loss.append(d_loss.item())
            generator_loss.append(g_loss.item())
    plt.plot(discriminator_loss)
    plt.title("the loss of discriminator")
    plt.savefig('Loss of D.jpg')
    plt.plot(generator_loss)
    plt.title("the loss of generator")
    plt.savefig('Loss of G.jpg')

    # batches_done = epoch * len(train_loader) + i
    # if batches_done % SAVE == 0:
    #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
