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

    def __init__(self, data_dir, n_in_mem=300):
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
        # updated loaded dictionary of data
        data = {}
        for key in ["imgs", "metos"]:
            path = [s for s in self.paths[key] if self.ids[i] in s][0]
            data[key] = dict(np.load(path).items())
        print("loading", i, end="\r")

        # rearrange into numpy arrays
        coords = np.stack([data["imgs"]["Lat"], data["imgs"]["Lon"]])
        imgs = np.stack([v for k, v in data["imgs"].items() if "Reflect" in k])
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
        return (coords, torch.Tensor(imgs), torch.Tensor(metos))
