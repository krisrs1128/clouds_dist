import torch
import torch.nn as nn

from src.unet_concise import UNet

# from src.cloud_unet import Discriminator
from src.res_discriminator import Discriminator


class GAN(nn.Module):
    def __init__(
        self,
        Cin,
        Cout,
        Cnoise,
        n_blocks=5,
        filter_factors=None,
        kernel_size=3,
        dropout=0.5,
        disc_size=64,
        device=None,
    ):
        super(GAN, self).__init__()

        self.g = UNet(
            Cin + Cnoise, Cout, n_blocks, filter_factors, kernel_size, dropout, device
        )
        self.d = Discriminator(Cout, disc_size, device=device)

        self.g.apply(self.init_weights)
        # self.d.apply(self.init_weights) # d has own init

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        x = self.g(x)
        return self.d(x)
