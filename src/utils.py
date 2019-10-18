from pathlib import Path
from addict import Dict
import yaml
import numpy as np


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
