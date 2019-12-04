#!/usr/bin/env python
from pathlib import Path
import datetime
import numpy as np
import re
import torch
from itertools import compress
from torch.utils.data import Dataset
from torchvision import transforms
from src.preprocessing import (
    ClipReflectance,
    CropInnerSquare,
    ReplaceNans,
    Standardize,
    SquashChannels,
    Resize,
    Quantize,
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
        self,
        data_dir,
        preprocessed_data_path=None,
        load_limit=-1,
        val_ids=set(),
        is_val=False,
        transform=None,
    ):
        super(EarthData).__init__()
        self.transform = transform
        self.preprocessed_data_path = preprocessed_data_path
        self.val_ids = val_ids
        self.is_val = is_val

        if preprocessed_data_path:
            data_dir = preprocessed_data_path

        val_ids = set(str(v) for v in val_ids)

        self.paths = {
            "real_imgs": {
                Path(g).stem.split("_")[1]: g for g in Path(data_dir).glob("imgs/*.npz")
            },
            "metos": {
                Path(g).stem.split("_")[1]: g
                for g in Path(data_dir).glob("metos/*.npz")
            },
        }

        self.paths = {
            "real_imgs": {
                k: v
                for k, v in self.paths["real_imgs"].items()
                if (is_val and k in val_ids) or (not is_val and k not in val_ids)
            },
            "metos": {
                k: v
                for k, v in self.paths["metos"].items()
                if (is_val and k in val_ids) or (not is_val and k not in val_ids)
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
    coords[np.isinf(coords)] = np.nan
    imgs = np.stack([v for k, v in data["real_imgs"].items() if "Reflect" in k])
    metos = np.concatenate(
        [
            data["metos"]["U"],
            data["metos"]["T"],
            data["metos"]["V"],
            data["metos"]["RH"],
            data["metos"]["Scattering_angle"].reshape(1, 256, 256),
            data["metos"]["TS"].reshape(1, 256, 256),
            coords,
        ]
    )
    return {"real_imgs": torch.Tensor(imgs), "metos": torch.Tensor(metos)}


class LowClouds(Dataset):
    """
    Low Clouds / Metereology Data

    Each index corresponds to one 128 x 128 low cloud image, along with 8
    meteorological variables. A separate metadata field stores parsed
    information from the filenames.

    Example
    -------
    >>> clouds = LowClouds("/scratch/sankarak/data/low_clouds/")
    """

    def __init__(
            self,
            data_dir,
            preprocessed_data_path=None,
            load_limit=-1,
            val_ids=set([]),
            is_val=False,
            transform=None
    ):
        super(LowClouds).__init__()

        # validation logic
        files = np.load(Path(data_dir, "files.npy"))
        self.ids = [Path(str(f)).name for f in files]
        self.ids = [re.search("([0-9]+.*)(?=_Block)", f).group() for f in self.ids]
        self.val_ids = val_ids
        self.is_val = is_val
        val_ids = set(str(v) for v in val_ids)

        if is_val:
            subset_ix = [s in val_ids for s in self.ids]
        else:
            subset_ix = [s not in val_ids for s in self.ids]

        # read data and subset
        self.ids = list(compress(self.ids, subset_ix))
        self.data = {
            "metos": np.load(Path(data_dir, "meto.npy")).transpose((1, 0))[subset_ix],
            "real_imgs": np.load(Path(data_dir, "train.npy")).transpose((2, 0, 1))[subset_ix],
        }

        # some metadata
        self.metadata = [{"id": s, "date": parse_dates(s)} for s in self.ids]
        self.transform = transform

        if load_limit != -1:
            self.ids = self.ids[:load_limit]
            self.data["metos"] = self.data["metos"][:load_limit]
            self.data["real_imgs"] = self.data["real_imgs"][:load_limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        data = {
            "metos": self.data["metos"][i],
            "real_imgs": self.data["real_imgs"][i],
        }

        if self.transform:
            data = self.transform(data)
        return data


def parse_dates(s):
    pattern = (
        "(20[0-9][0-9])([0-9][0-9])([0-9]+).([0-9][0-9])([0-9][0-9])"
    )  # regexr.com/4ponn
    groups = [int(g) for g in re.search(pattern, s).groups()]
    try:
        groups[1], groups[2] = groups[2], groups[1]  # month and day are swapped
        result = datetime.datetime(*groups)
    except:
        result = np.nan

    return result


def get_nan_value(transfs):
    nan_value = "raw"
    for t in transfs:
        if t.__class__.__name__ == "Standardize":
            nan_value = "Standardize"
        elif t.__class__.__name__ == "Quantize":
            nan_value = "Quantize"
    return nan_value


def get_transforms(opts):
    transfs = []
    if opts.data.crop_to_inner_square:
        transfs += [CropInnerSquare()]
    transfs += [Resize(256)]
    if opts.data.squash_channels:
        transfs += [SquashChannels()]
        assert (
            opts.model.Cin == 8
        ), "using squash_channels, Cin should be 8 not {}".format(opts.model.Cin)

    if opts.data.clip_reflectance and opts.data.clip_reflectance > 0:
        transfs += [ClipReflectance(opts.data.clip_reflectance)]
    if opts.data.noq:
        transfs += [Quantize()]
    elif opts.data.preprocessed_data_path is None and opts.data.with_stats:
        transfs += [Standardize()]
    nan_value = get_nan_value(transfs)
    transfs += [ReplaceNans(nan_value)]

    return transfs


def get_loader(opts, transfs=None, stats=None):
    if stats is not None:

        # make sure not to quantize and standardize at the same time
        stand_or_quant = False
        for t in transfs:
            if "Standardize" in str(t.__class__) or "Quantize" in str(t.__class__):
                if stand_or_quant:
                    raise ValueError(
                        "cannot perform both quantization AND standardization"
                    )

                t.set_stats(stats)
                stand_or_quant = True

            if "ReplaceNans" in str(t.__class__):
                t.set_stats(stats)

    dataset_args = {
        "data_dir": opts.data.path,
        "preprocessed_data_path": opts.data.preprocessed_data_path,
        "load_limit": opts.data.load_limit or -1,
        "transform": transforms.Compose(transfs),
    }

    if opts.data.cloud_type not in {"global", "local"}:
        raise ValueError(
            "Cloud type must be either 'global' or 'local', got {}".format(
                opts.data.cloud_type
            )
        )

    if opts.data.cloud_type == "global":
        trainset = EarthData(
            **dataset_args, is_val=False, val_ids=opts.val.get("val_ids", [])
        )
        valset = EarthData(
            **dataset_args, is_val=True, val_ids=opts.val.get("val_ids", [])
        )
    else:
        trainset = LowClouds(
            **dataset_args, is_val=False, val_ids=opts.val.get("val_ids", [])
        )
        valset = LowClouds(
            **dataset_args, is_val=True, val_ids=opts.val.get("val_ids", [])
        )

    transforms_string = ""
    if transfs:
        transforms_string += " -> ".join([t.__class__.__name__ for t in transfs])

    return (
        torch.utils.data.DataLoader(
            trainset,
            batch_size=opts.train.batch_size,
            shuffle=True,
            # sampler=torch.utils.data.SubsetRandomSampler([0]*45),
            num_workers=opts.data.get("num_workers", 3),
        ),
        torch.utils.data.DataLoader(
            valset,
            batch_size=opts.train.batch_size,
            shuffle=False,
            # sampler=torch.utils.data.SubsetRandomSampler([0]*5),
            num_workers=opts.data.get("num_workers", 3),
        ),
        transforms_string,
    )
