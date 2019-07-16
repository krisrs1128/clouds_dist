#!/usr/bin/env python
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import os.path
import re
import torch


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

    def __getitem__(self, ix):
        # updated loaded dictionary of data
        if ix not in self.cur_ix:
            start = ix - ix % self.n_in_mem
            self.cur_ix = range(start, start + self.n_in_mem)

            # load imgs / metos one by one
            self.subsample = {}
            for i in self.cur_ix:
                data = {}
                for key in ["imgs", "metos"]:
                    path = [s for s in self.paths[key] if self.ids[i] in s][0]
                    data[key] = dict(np.load(path).items())

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

                self.subsample[i] = (coords, torch.Tensor(imgs), torch.Tensor(metos))

        return self.subsample[ix]
