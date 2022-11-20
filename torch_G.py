import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import randn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from pytorch_model_summary import summary


# fake class label 0
# true class label 1

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.name = "G"
        self.latent_dim = 100
        self.batch_size = 64
        # first hidden layer to
        self.h1 = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.batch_size * 256 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # upsample n*n -> 2n * 2n
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            # 128 is the channel size, should match with the real image data loader,H = (H1 - 1)*stride + HF - 2*padding
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1),  # 3 means the RGB channel
            nn.Tanh()
        )
        self.transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])

    def forward(self, x):
        out = self.h1(x)
        out = out.view(self.batch_size, 256, 4, 4)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up2(out)  # 128*32*32
        out = self.output(out)  # TODO 是否需要normalize到[-1,1]要！因为把real的也用transform这样转换了
        # normalize to [-1,1]
        # out = self.transform_norm(out)
        return out

    # no need, implement in torch_GAN main
    def generate_latent_points(self):
        return torch.normal(mean=0, std=1, size=(self.batch_size, self.latent_dim))

    # no need
    def generate_fake_samples(self):
        latent = self.generate_latent_points()
        x = self.forward(latent)
        y = torch.zeros(self.batch_size, 1)
        return x, y

    # no need
    def plot(self):
        # plot the generated samples
        x, _ = self.generate_fake_samples()
        for i in range(self.batch_size):
            # define subplot
            plt.subplot(7, 7, 1 + i)  # when the batch_size = 49, choose 7
            # turn off axis labels
            plt.axis('off')
            # plot single image
            plt.imshow(x[i])
        # show the figure
        plt.show()


if __name__ == "__main__":
    g = G()
    print(summary(g, torch.zeros(100)))
