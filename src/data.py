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
        self.subsample = {}
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

        stand_or_quant = (
            False
        )  # make sure not to quantize and standarize at the same time
        for t in transfs:
            if "Standardize" in str(t.__class__) or "Quantize" in str(t.__class__):
                assert not stand_or_quant, (
                    "cannot perform quantization and"
                    + " standardization at the same time!"
                )

                t.set_stats(stats)
                stand_or_quant = True

            if "ReplaceNans" in str(t.__class__):
                t.set_stats(stats)

    trainset = EarthData(
        opts.data.path,
        preprocessed_data_path=opts.data.preprocessed_data_path,
        load_limit=opts.data.load_limit or -1,
        transform=transforms.Compose(transfs),
        is_val=False,
        val_ids=opts.val.get("val_ids", []),
    )

    valset = EarthData(
        opts.data.path,
        preprocessed_data_path=opts.data.preprocessed_data_path,
        load_limit=opts.data.load_limit or -1,
        transform=transforms.Compose(transfs),
        is_val=True,
        val_ids=opts.val.get("val_ids", []),
    )

    transforms_string = " -> ".join([t.__class__.__name__ for t in transfs])

    return (
        torch.utils.data.DataLoader(
            trainset,
            batch_size=opts.train.batch_size,
            shuffle=True,
            num_workers=opts.data.get("num_workers", 3),
        ),
        torch.utils.data.DataLoader(
            valset,
            batch_size=opts.train.batch_size,
            shuffle=False,
            num_workers=opts.data.get("num_workers", 3),
        ),
        transforms_string,
    )


{
    "real_imgs": {
        "20160927143257": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160927143257_02_256x256.npz"
        ),
        "20160515165826": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160515165826_02_256x256.npz"
        ),
        "20160615232154": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160615232154_02_256x256.npz"
        ),
        "20160218203424": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160218203424_02_256x256.npz"
        ),
        "20160718033415": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160718033415_02_256x256.npz"
        ),
        "20160530100701": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160530100701_02_256x256.npz"
        ),
        "20160105040318": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160105040318_02_256x256.npz"
        ),
        "20160307001751": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160307001751_02_256x256.npz"
        ),
        "20161010020554": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161010020554_02_256x256.npz"
        ),
        "20160715141008": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160715141008_02_256x256.npz"
        ),
        "20160813161144": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160813161144_02_256x256.npz"
        ),
        "20160716030611": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160716030611_02_256x256.npz"
        ),
        "20161001043122": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161001043122_02_256x256.npz"
        ),
        "20160510174747": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160510174747_02_256x256.npz"
        ),
        "20160817023808": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160817023808_02_256x256.npz"
        ),
        "20161012111530": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161012111530_02_256x256.npz"
        ),
        "20161031091806": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161031091806_02_256x256.npz"
        ),
        "20160119111530": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160119111530_02_256x256.npz"
        ),
        "20160110001752": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160110001752_02_256x256.npz"
        ),
        "20160511143806": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160511143806_02_256x256.npz"
        ),
        "20160629002712": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160629002712_02_256x256.npz"
        ),
        "20161022114337": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161022114337_02_256x256.npz"
        ),
        "20160925150058": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160925150058_02_256x256.npz"
        ),
        "20160618203334": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160618203334_02_256x256.npz"
        ),
        "20160710165828": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160710165828_02_256x256.npz"
        ),
        "20160215144212": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160215144212_02_256x256.npz"
        ),
        "20160224073924": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160224073924_02_256x256.npz"
        ),
        "20160723161141": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160723161141_02_256x256.npz"
        ),
        "20160827093857": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160827093857_02_256x256.npz"
        ),
        "20160630015122": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160630015122_02_256x256.npz"
        ),
        "20160727221627": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160727221627_02_256x256.npz"
        ),
        "20160723041138": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160723041138_02_256x256.npz"
        ),
        "20160710222547": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160710222547_02_256x256.npz"
        ),
        "20160302182742": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160302182742_02_256x256.npz"
        ),
        "20160612134207": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160612134207_02_256x256.npz"
        ),
        "20160408044041": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160408044041_02_256x256.npz"
        ),
        "20160508113106": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160508113106_02_256x256.npz"
        ),
        "20161104134057": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161104134057_02_256x256.npz"
        ),
        "20160529201452": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160529201452_02_256x256.npz"
        ),
        "20161107200625": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161107200625_02_256x256.npz"
        ),
        "20160512160220": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160512160220_02_256x256.npz"
        ),
        "20161107181822": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161107181822_02_256x256.npz"
        ),
        "20160803044903": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160803044903_02_256x256.npz"
        ),
        "20160629232151": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160629232151_02_256x256.npz"
        ),
        "20160714004554": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160714004554_02_256x256.npz"
        ),
        "20161214145136": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20161214145136_02_256x256.npz"
        ),
        "20160821024729": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160821024729_02_256x256.npz"
        ),
        "20160708073721": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160708073721_02_256x256.npz"
        ),
        "20160812105346": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160812105346_02_256x256.npz"
        ),
        "20160206043121": PosixPath(
            "/Tmp/schmidtv/slurm-266782/imgs/img_20160206043121_02_256x256.npz"
        ),
    },
    "metos": {
        "20160515165826": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160515165826_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160206043121": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160206043121_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160510174747": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160510174747_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161012111530": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161012111530_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161031091806": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161031091806_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160215144212": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160215144212_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160710165828": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160710165828_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160714004554": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160714004554_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160710222547": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160710222547_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161107181822": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161107181822_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160615232154": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160615232154_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160508113106": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160508113106_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161010020554": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161010020554_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160708073721": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160708073721_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160218203424": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160218203424_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160821024729": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160821024729_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160512160220": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160512160220_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160827093857": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160827093857_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160511143806": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160511143806_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161214145136": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161214145136_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160612134207": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160612134207_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161001043122": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161001043122_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160925150058": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160925150058_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160813161144": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160813161144_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160530100701": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160530100701_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160715141008": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160715141008_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161104134057": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161104134057_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160723041138": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160723041138_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160817023808": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160817023808_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160723161141": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160723161141_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160408044041": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160408044041_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160630015122": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160630015122_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160224073924": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160224073924_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161107200625": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161107200625_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160307001751": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160307001751_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160927143257": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160927143257_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160110001752": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160110001752_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160105040318": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160105040318_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160629232151": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160629232151_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160119111530": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160119111530_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160529201452": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160529201452_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160716030611": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160716030611_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160727221627": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160727221627_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160812105346": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160812105346_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160718033415": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160718033415_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160302182742": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160302182742_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160618203334": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160618203334_02_256x256_Collocated_MERRA2.npz"
        ),
        "20161022114337": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20161022114337_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160629002712": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160629002712_02_256x256_Collocated_MERRA2.npz"
        ),
        "20160803044903": PosixPath(
            "/Tmp/schmidtv/slurm-266782/metos/img_20160803044903_02_256x256_Collocated_MERRA2.npz"
        ),
    },
}

