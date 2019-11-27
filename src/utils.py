#!/usr/bin/env python
from addict import Dict
from pathlib import Path
from src.cluster_utils import env_to_path
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F
import wandb
import yaml


def load_conf(path):
    path = Path(path).resolve()
    print("Loading conf from", str(path))
    with open(path, "r") as stream:
        try:
            return Dict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def sample_param(sample_dict):
    """sample a value (hyperparameter) from the instruction in the
    sample dict:
    {
        "sample": "range | list",
        "from": [min, max, step] | [v0, v1, v2 etc.]
    }
    if range, as np.arange is used, "from" MUST be a list, but may contain
    only 1 (=min) or 2 (min and max) values, not necessarily 3

    Args:
        sample_dict (dict): instructions to sample a value

    Returns:
        scalar: sampled value
    """
    if "sample" not in sample_dict:
        return sample_dict
    if sample_dict["sample"] == "range":
        value = np.random.choice(np.arange(*sample_dict["from"]))
    elif sample_dict["sample"] == "list":
        value = np.random.choice(sample_dict["from"])
    elif sample_dict["sample"] == "uniform":
        value = np.random.uniform(*sample_dict["from"])
    else:
        raise ValueError("Unknonw sample type in dict " + str(sample_dict))
    return value


def merge_defaults(extra_opts, conf_path):
    print("Loading params from", conf_path)
    result = load_conf(conf_path)
    for group in ["model", "train", "data"]:
        if group in extra_opts:
            for k, v in extra_opts[group].items():
                result[group][k] = v
    for group in ["model", "train", "data"]:
        for k, v in result[group].items():
            if isinstance(v, dict):
                v = sample_param(v)
            result[group][k] = v

    return Dict(result)


def to_0_1(arr_or_tensor):
    """scales a tensor/array to [0, 1] values:
    (x - min(x)) / (max(x) - min(x))

    Args:
        arr_or_tensor (torch.Tensor or np.array): input tensor to scale

    Returns:
        torch.Tensor or np.array: scaled tensor
    """

    return (arr_or_tensor - arr_or_tensor.min()) / (
        arr_or_tensor.max() - arr_or_tensor.min()
    )


def loss_hinge_dis(dis_fake, dis_real):
    # This version returns a single loss
    # from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
    loss = torch.mean(F.relu(1.0 - dis_real))
    loss += torch.mean(F.relu(1.0 + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    # from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
    loss = -torch.mean(dis_fake)
    return loss


def weighted_mse_loss(input, target):
    # from https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254/2
    out = (input - target) ** 2
    weights = (input.sum(1) != 0).to(torch.float32)
    weights = weights.unsqueeze(1).expand_as(out) / weights.sum()
    out = out * weights
    loss = out.sum()
    return loss


def get_opts(conf_path):
    if not Path(conf_path).exists():
        conf_name = conf_path
        if not conf_name.endswith(".yaml"):
            conf_name += ".yaml"
        conf_path = Path(__file__).parent.parent / "shared" / conf_name
        assert conf_path.exists()

    return merge_defaults({"model": {}, "train": {}, "data": {}}, conf_path)


def check_data_dirs(opts):
    opts.data.path = env_to_path(opts.data.path)
    opts.data.preprocessed_data_path = env_to_path(opts.data.preprocessed_data_path)
    opts.data.original_path = env_to_path(opts.data.original_path)
    if not opts.data.path:
        opts.data.path = opts.data.original_path
        print(
            "check_dirs():\n",
            "No opts.data.path, fallback to opts.data.original_path: {}".format(
                opts.data.original_path
            ),
        )

    print("Loading data from ", str(opts.data.path))
    assert Path(opts.data.path).exists(), "{} does not exist".format(
        str(Path(opts.data.path))
    )
    assert (Path(opts.data.path) / "imgs").exists(), "{} does not exist".format(
        str(Path(opts.data.path) / "imgs")
    )
    assert (Path(opts.data.path) / "metos").exists(), "{} does not exist".format(
        str(Path(opts.data.path) / "metos")
    )
    return opts


def cpu_images(input_tensor, real_img, generated_img):
    imgs = []
    for i in range(input_tensor.shape[0]):
        # concatenate verticaly:
        # [3 metos, generated clouds, ground truth clouds]
        img = torch.cat(
            (
                to_0_1(input_tensor[i, 22:25]),
                to_0_1(generated_img[i]),
                to_0_1(real_img[i]),
            ),
            1,
        )
        img_cpu = img.cpu().clone().detach().numpy()
        imgs.append(np.swapaxes(img_cpu, 0, 2))

    return imgs


def record_images(imgs, store_images, exp, imgdir, step, nb_images, val_epoch):
    for i, im in enumerate(imgs):
        im_caption = f"imgs_{step}_{nb_images + i}"
        if store_images:
            plt.imsave(str(imgdir / im_caption) + ".png", im)
        if exp:
            try:
                wandb.log(
                    {
                        "inference": [wandb.Image(im, caption=im_caption)],
                        "index_in_batch": i,
                        "sample": nb_images + i,
                    },
                    step=step,
                )
            except Exception as e:
                print(f"\n{e}\n")


def subset_keys(D0, patterns):
    """
    Subset Dictionary using Regexes

    We use this when we want to partially load a state dict, but want to
    specify the parameters to keep using regexes (rather than a list of all
    parameter names).

    :param D0: A dictionary whose elements we want to subset.
    :param patterns: A list of regex strings, specifying which keys to subset
      down to. If a key matches any of the regexes, it is included in the
      output dictionary.
    """
    D = D0.copy()
    for k in list(D.keys()):
        omit = True
        for pattern in patterns:
            if re.compile(pattern).match(k):
                omit = False
                continue
        if omit:
            D.pop(k)

    return D
