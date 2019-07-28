#!/usr/bin/env python
import torch
from pathlib import Path
import torch.nn as nn


class UNetModule(nn.Module):
    """
    One of the "triple layer" blocks in https://arxiv.org/pdf/1505.04597.pdf
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, padding=1)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, padding=1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x):
        layers = nn.Sequential(
            self.conv1, self.bn, self.activation,
            self.conv2, self.bn, self.activation
        )
        return layers(x)


class UNet(nn.Module):
    """
    Encoder-Decoder with Skip Connections

    This is a vanilla U-Net, with batch norm added in. It assumes that the
    output mask y is normalized between 0 and 1.

    Example
    -------
    >>> model = UNet()
    >>> x = torch.randn(1, 3, 512, 512)
    >>> y = model(x)
    """

    def __init__(self):
        super().__init__()
        self.filter_factors = [1, 2, 4, 8, 16]
        self.n_channels = 3
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        input_sizes = [32 * s for s in self.filter_factors]

        # link up all the encoder and decoder components
        self.down = [UNetModule(self.n_channels, input_sizes[0])]
        self.up = []
        for i in range(len(input_sizes) - 1):
            self.down.append(UNetModule(input_sizes[i], input_sizes[i + 1]))
            self.up.append(UNetModule(input_sizes[i] + input_sizes[i + 1], input_sizes[i]))

        self.conv_final = nn.Conv2d(input_sizes[0], self.n_channels, 1)

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
        return torch.sigmoid(out)
