#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class unet_block(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        kernel_size=3,
        dropout=0.5,
        submodule=None,
        outermost=False,
    ):
        super(unet_block, self).__init__()
        self.outermost = outermost
        self.sub = submodule

        P = int((kernel_size - 1) / 2)

        model = [
            nn.Conv2d(outer_nc, outer_nc, kernel_size=kernel_size, stride=1, padding=P),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(outer_nc, outer_nc, kernel_size=kernel_size, stride=1, padding=P),
            nn.LeakyReLU(0.2, True),
        ]
        self.input = nn.Sequential(*model)

        model = [
            nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        self.down = nn.Sequential(*model)

        if submodule is None:
            model = [
                nn.Conv2d(
                    inner_nc, inner_nc, kernel_size=kernel_size, stride=1, padding=P
                ),
                nn.LeakyReLU(0.2, True),
            ]
            self.sub = nn.Sequential(*model)

        model = [
            nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout),
        ]
        self.up = nn.Sequential(*model)

        model = [
            nn.Conv2d(
                2 * outer_nc, outer_nc, kernel_size=kernel_size, stride=1, padding=P
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(outer_nc, outer_nc, kernel_size=kernel_size, stride=1, padding=P),
            nn.ReLU(True),
        ]
        self.output = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down(x1)
        x3 = self.sub(x2)
        x4 = self.up(x3)
        x5 = torch.cat([x1, x4], dim=1)
        return self.output(x5)


class unet(nn.Module):
    def __init__(self, Cin, Cout, n_channels, nblocks, kernel_size, dropout):
        super(unet, self).__init__()

        inconv = nn.Conv2d(Cin, n_channels, kernel_size=1, stride=1, padding=0)
        self.input = nn.Sequential(inconv)

        submodule = None
        for i in range(nblocks - 1):
            submodule = unet_block(
                n_channels,
                n_channels,
                kernel_size,
                dropout,
                submodule=submodule,
                outermost=False,
            )
        self.ublock = unet_block(
            n_channels,
            n_channels,
            kernel_size,
            dropout,
            submodule=submodule,
            outermost=True,
        )

        outconv = nn.Conv2d(n_channels, Cout, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sequential(outconv)

    def forward(self, x):
        x = self.input(x)
        x = self.ublock(x)
        x = self.output(x)
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self, Cin, n_channels=16, nlevels=4, device=None):
        super(Discriminator, self).__init__()

        kernel_size = 4
        P = int((kernel_size - 2) / 2)
        Cin = Cin
        Cout = n_channels

        model = []
        for i in range(nlevels):
            model += [
                nn.Conv2d(Cin, Cout, kernel_size, stride=2, padding=P),
                nn.LeakyReLU(0.2, True),
            ]
            Cin = Cout
            Cout *= 2

        model += [nn.Conv2d(Cin, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)
        if device:
            self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
