import torch.nn as nn

from src.unet import UNet
from src.res_discriminator import Discriminator, MultiDiscriminator


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
        bottleneck_dim=27,
        device=None,
        multi_disc=False,
        use_leaky=False,
    ):
        super(GAN, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.g = UNet(
            Cin + Cnoise,
            Cout,
            n_blocks,
            filter_factors,
            kernel_size,
            dropout,
            bottleneck_dim,
            device,
            use_leaky,
        )
        self.d = (
            Discriminator(Cout, disc_size, device=device)
            if not multi_disc
            else MultiDiscriminator(Cout, device=device)
        )

        self.g.apply(self.init_weights)
        # self.d.apply(self.init_weights) # d has own init

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.uniform_(m.bias, -0.1, 0.1)
