#!/usr/bin/env python
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocessing import (
    ClipReflectance,
    CropInnerSquare,
    ReplaceNans,
    Standardize,
    SquashChannels,
    Rescale,
)


class EarthData(Dataset):
    """
    Earth Clouds / Metereology Data

    Each index corresponds to one timepoint in the clouds and meteorology
    simulation. The returned tuple is (coords, imgs, metos).

    :param data_dir: The path containing the imgs/ and metos/ subdirectories.

    Example
    -------
    >>> from torch.utils.data import DataLoader
    >>> earth = EarthData("/data/")
    >>> loader = DataLoader(earth)
    >>> for i, elem in enumerate(loader):
    >>>    coords, x, y = elem
    >>>    print(x.shape)
    """

    def __init__(
        self, data_dir, preprocessed_data_path=None, load_limit=-1, transform=None
    ):
        super(EarthData).__init__()
        self.subsample = {}
        self.transform = transform
        self.preprocessed_data_path = preprocessed_data_path

        if preprocessed_data_path:
            data_dir = preprocessed_data_path

        self.paths = {
            "real_imgs": {
                Path(g).stem.split("_")[1]: g for g in Path(data_dir).glob("imgs/*.npz")
            },
            "metos": {
                Path(g).stem.split("_")[1]: g
                for g in Path(data_dir).glob("metos/*.npz")
            },
        }
        self.ids = list(self.paths["real_imgs"].keys())[:load_limit]

        # ------------------------------------
        # ----- Infer Data Size for Unet -----
        # ------------------------------------
        data = {}
        for key in ["real_imgs", "metos"]:
            path = self.paths[key][self.ids[0]]
            if self.preprocessed_data_path:
                data[key] = np.load(path)[key]
            else:
                data[key] = dict(np.load(path).items())
        if self.preprocessed_data_path is None:
            data = process_sample(data)
        self.toy_data = data
        self.metos_shape = tuple(data["metos"].shape)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        data = {}
        id = self.ids[i]
        for key in ["real_imgs", "metos"]:
            path = self.paths[key][id]
            if self.preprocessed_data_path:
                data[key] = np.load(path)[key]
            else:
                data[key] = dict(np.load(path).items())

        if self.preprocessed_data_path is None:
            data = process_sample(data)

        if self.transform:
            data = self.transform(data)
        return data


def process_sample(data):
    # rearrange into numpy arrays
    coords = np.stack([data["real_imgs"]["Lat"], data["real_imgs"]["Lon"]])
    imgs = np.stack([v for k, v in data["real_imgs"].items() if "Reflect" in k])
    metos = np.concatenate(
        [
            data["metos"]["U"],
            data["metos"]["T"],
            data["metos"]["V"],
            data["metos"]["RH"],
            data["metos"]["Scattering_angle"].reshape(1, 256, 256),
            data["metos"]["TS"].reshape(1, 256, 256),
            coords.reshape(2, 256, 256),
        ]
    )
    return {"real_imgs": torch.Tensor(imgs), "metos": torch.Tensor(metos)}


def get_transforms(opts):
    transfs = []
    if opts.data.crop_to_inner_square:
        transfs += [CropInnerSquare()]
    transfs += [Rescale(256)]
    if opts.data.squash_channels:
        transfs += [SquashChannels()]
        assert (
            opts.model.Cin == 8
        ), "using squash_channels, Cin should be 8 not {}".format(opts.model.Cin)
    if opts.data.clip_reflectance and opts.data.clip_reflectance > 0:
        transfs += [ClipReflectance(opts.data.clip_reflectance)]
    if opts.data.preprocessed_data_path is None and opts.data.with_stats:
        transfs += [Standardize()]
    transfs += [ReplaceNans()]

    return transfs


def get_loader(opts, transfs=None, stats=None):

    if stats is not None:
        for t in transfs:
            if "Standardize" in str(t.__class__):
                t.set_stats(stats)

    trainset = EarthData(
        opts.data.path,
        preprocessed_data_path=opts.data.preprocessed_data_path,
        load_limit=opts.data.load_limit or -1,
        transform=transforms.Compose(transfs),
    )

    transforms_string = " -> ".join([t.__class__.__name__ for t in transfs])

    return (
        torch.utils.data.DataLoader(
            trainset,
            batch_size=opts.train.batch_size,
            shuffle=True,
            num_workers=opts.data.get("num_workers", 3),
        ),
        transforms_string,
    )
