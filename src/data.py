#!/usr/bin/env python
import os.path
import re
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class EarthData(Dataset):
    """
    Earth Clouds / Metereology Data

    Each index corresponds to one timepoint in the clouds and meteorology
    simulation. The returned tuple is (coords, imgs, metos).

    :param data_dir: The path containing the imgs/ and metos/ subdirectories.
    :n_in_mem: The number of samples to load (into CPU memory) at a time.

    Example
    -------
    >>> data = EarthData("/data/")
    >>> x, y = data[0]
    """

    def __init__(self, data_dir, n_in_mem=50):
        super(EarthData).__init__()
        self.n_in_mem = n_in_mem
        self.cur_ix = []
        self.subsample = {}

        self.paths = {
            "imgs": glob(os.path.join(data_dir, "imgs", "*.npz")),
            "metos": glob(os.path.join(data_dir, "metos", "*.npz")),
        }

        self.ids = [re.search("[0-9]+", s).group() for s in self.paths["imgs"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # if already available, return
        if i in self.subsample.keys():
            return self.subsample[i]

        # otherwise load next n_in_mem images
        subsample = {}
        for j in range(i, i + self.n_in_mem):
            data = {}
            for key in ["imgs", "metos"]:
                path = [s for s in self.paths[key] if self.ids[j] in s][0]
                data[key] = dict(np.load(path).items())
                print("loading {} {}".format(j, key))

            subsample[j] = process_sample(data)

        self.subsample = subsample
        print(len(self.subsample))
        return self.subsample[j]


def process_sample(data):
    # rearrange into numpy arrays
    coords = np.stack([data["imgs"]["Lat"], data["imgs"]["Lon"]])
    imgs = np.stack([v for k, v in data["imgs"].items() if "Reflect" in k])
    imgs[np.isnan(imgs)] = 0.
    imgs[np.isinf(imgs)] = 0.
    metos = np.concatenate(
        [
            data["metos"]["U"],
            data["metos"]["T"],
            data["metos"]["V"],
            data["metos"]["RH"],
            data["metos"]["Scattering_angle"].reshape(1, 256, 256),
            data["metos"]["TS"].reshape(1, 256, 256),
        ]
    )
    metos[np.isnan(metos)] = 0.
    metos[np.isinf(metos)] = 0.
    return (coords, torch.Tensor(imgs), torch.Tensor(metos))
