#!/usr/bin/env python
import os.path
import re
from glob import glob
import gc

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
    >>> from torch.utils.data import DataLoader
    >>> earth = EarthData("/data/")
    >>> loader = DataLoader(earth)
    >>> for i, elem in enumerate(loader):
    >>>    coords, x, y = elem
    >>>    print(x.shape)
    """

    def __init__(self, data_dir, preprocessed_data_path=None, n_in_mem=500, load_limit=-1, transform=None):
        super(EarthData).__init__()
        self.n_in_mem = n_in_mem
        self.subsample = {}
        self.transform = transform
        self.preprocessed_data_path = preprocessed_data_path

        if preprocessed_data_path:
            data_dir = preprocessed_data_path

        self.paths = {
            "real_imgs": glob(os.path.join(data_dir, "imgs", "*.npz")),
            "metos": glob(os.path.join(data_dir, "metos", "*.npz")),
        }
        self.ids = [re.search("[0-9]+", s).group() for s in self.paths["real_imgs"]][
            :load_limit
        ]

        print("Loading elements (n_in_mem): ", n_in_mem)

        # ------------------------------------
        # ----- Infer Data Size for Unet -----
        # ------------------------------------
        data = {}
        for key in ["real_imgs", "metos"]:
            path = [s for s in self.paths[key] if self.ids[0] in s][0]
            if self.preprocessed_data_path:
                data[key] = np.load(path)[key]
            else:
                data[key] = dict(np.load(path).items())
        if self.preprocessed_data_path is None:
            data = process_sample(data)
        self.metos_shape = tuple(data["metos"].shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # if already available, return
        if i in self.subsample.keys():
            return self.subsample[i]

        # otherwise load next n_in_mem images
        self.subsample = {}
        gc.collect()

        subsample_end = min(i + self.n_in_mem, self.__len__())
        for j in range(i, subsample_end):
            data = {}
            for key in ["real_imgs", "metos"]:
                path = [s for s in self.paths[key] if self.ids[j] in s][0]
                if self.preprocessed_data_path:
                    data[key] = np.load(path)[key]
                else:
                    data[key] = dict(np.load(path).items())

            if self.preprocessed_data_path is None:
                data = process_sample(data)

            self.subsample[j] = data
            if self.transform:
                self.subsample[j] = self.transform(self.subsample[j])

        return self.subsample[i]


def process_sample(data):
    # rearrange into numpy arrays
    coords = np.stack([data["real_imgs"]["Lat"], data["real_imgs"]["Lon"]])
    imgs = np.stack([v for k, v in data["real_imgs"].items() if "Reflect" in k])
    imgs[np.isnan(imgs)] = 0.0
    imgs[np.isinf(imgs)] = 0.0
    metos = np.concatenate(
        [
            data["metos"]["U"],
            data["metos"]["T"],
            data["metos"]["V"],
            data["metos"]["RH"],
            data["metos"]["Scattering_angle"].reshape(1, 256, 256),
            data["metos"]["TS"].reshape(1, 256, 256),
            coords.reshape(2, 256, 256)
        ]
    )
    metos[np.isnan(metos)] = 0.0
    metos[np.isinf(metos)] = 0.0
    return {"real_imgs": torch.Tensor(imgs),
        "metos": torch.Tensor(metos),
    }
