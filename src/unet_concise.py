#!/usr/bin/env python
import torch
from pathlib import Path
import torch.nn as nn
import numpy as np


class UNetModule(nn.Module):
    """
    One of the "triple layer" blocks in https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, n_in, n_out, kernel_size=3, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        layers = nn.Sequential(
            self.conv1,
            self.bn,
            self.activation,
            self.drop,
            self.conv2,
            self.bn,
            self.activation,
            self.drop,
        )
        return layers(x)


class UNet(nn.Module):
    """
    Encoder-Decoder with Skip Connections

    This is a vanilla U-Net, with batch norm added in. It assumes that the
    output mask y is normalized between 0 and 1.

    Example
    -------
    >>> model = UNet(42, 3)
    >>> x = torch.randn(1, 42, 216, 216)
    >>> y = model(x)
    """

    def __init__(
        self,
        Cin,
        Cout,
        n_blocks=4,
        filter_factors=None,
        kernel_size=3,
        dropout=0.5,
        bottleneck_dim=27,
        device=None,
    ):
        super().__init__()
        if not filter_factors:
            self.filter_factors = list(2 ** np.arange(n_blocks))
        else:
            self.filter_factors = filter_factors

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        input_sizes = [bottleneck_dim * s for s in self.filter_factors]

        # link up all the encoder and decoder components
        self.down = [UNetModule(Cin, input_sizes[0], kernel_size, dropout)]
        self.up = []
        for i in range(len(input_sizes) - 1):
            self.down.append(
                UNetModule(input_sizes[i], input_sizes[i + 1], kernel_size, dropout)
            )
            self.up.append(
                UNetModule(
                    input_sizes[i] + input_sizes[i + 1],
                    input_sizes[i],
                    kernel_size,
                    dropout,
                )
            )

        self.conv_final = nn.Conv2d(input_sizes[0], Cout, 1)
        if device:
            self.pool = self.pool.to(device)
            self.upsample = self.upsample.to(device)
            self.down = [d.to(device) for d in self.down]
            self.up = [u.to(device) for u in self.up]

    def forward(self, x):
        # encoder pass
        down_features = []
        for i, down in enumerate(self.down):
            f = x if i == 0 else self.pool(down_features[-1])
            down_features.append(down(f))

        # decoder pass
        out = down_features[-1]
        for i, up in enumerate(reversed(self.up)):
            f = torch.cat([self.upsample(out), down_features[-(i + 2)]], 1)
            out = up(f)

        out = self.conv_final(out)
        return torch.tanh(out)
        # return torch.sigmoid(out)
