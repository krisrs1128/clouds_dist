import torch
import torch.nn as nn

from src.cloud_unet import Discriminator, unet
from src.unets import UNet


class GAN(nn.Module):
    def __init__(self, Cin, Cout, n_channels, n_blocks, kernel_size, dropout):
        super(GAN, self).__init__()

        self.g = unet(Cin, Cout, n_channels, n_blocks, kernel_size, dropout)
        self.d = Discriminator(Cout, n_channels=16, nlevels=4)

        self.g.apply(self.init_weights)
        self.d.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        x = self.g(x)
        return self.d(x)
