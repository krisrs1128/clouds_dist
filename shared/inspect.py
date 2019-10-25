#!/usr/bin/env python
"""
Utilities to Study Model Results
"""
import torch
import numpy as np
import seaborn as sns
import argparse
from src.data import get_loader, get_transforms
import src.gan as gan
from src.utils import get_opts
from src.stats import get_stats
import pathlib


def infer(model, loader, out_dir=None, M=10):
    """
    Predictions on a Subset

    :param model: A torch model nn.Module object, to use for making
      predictions.
    :param samples: A torch dataloader object.
    :param M: The maximum number of batches to go through.
    """
    result = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    M = min(len(loader.dataset) / loader.batch_size, M)

    for m, batch in enumerate(loader):
        if m > M: break
        print(f"{m}/{M}")
        y_hat = model.g(batch["metos"].to(device))
        result.append(y_hat.detach().cpu())
    return torch.cat(result)


def histogram(x, sample_frac=0.3, out_path=None):
    """
    Values from Random Tensor Indices
    """
    if not out_path:
        out_path = pathlib.Path.cwd()

    x = x.numpy().flatten()
    indices = np.random.choice(len(x), int(len(x) * sample_frac))
    sns.distplot(x[indices]).figure.savefig(pathlib.Path(out_path, "figure.png"))


def loader_from_run(opts_path):
    """
    Get Loader (with transforms) from Experiment
    """
    opts = get_opts(opts_path)
    opts["data"]["path"] = opts["data"]["original_path"]

    # get transforms, stats, and return the loader
    transfs = get_transforms(opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = get_stats(opts, device, transfs)
    loader, _ = get_loader(opts, transfs, stats)
    return loader


def model_from_run(opts_path, checkpoints_dir, model_name):
    """
    Get Model from Experiment
    """
    opts = get_opts(opts_path)
    opts["data"]["path"] = opts["data"]["original_path"]
    model_path = pathlib.Path(checkpoints_dir, model_name)

    state = torch.load(model_path)["state_dict"]
    model = gan.GAN(
        opts["model"]["Cin"],
        opts["model"]["Cout"],
        opts["model"]["Cnoise"],
        bottleneck_dim=opts["model"]["bottleneck_dim"]
    )
    model.load_state_dict(state)
    return model


if __name__ == '__main__':
    # get file arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf_path",
        type=str,
        default= "/scratch/sankarak/clouds/extragrad-run--8/run_0/extragrad-run.yaml",
        help="The full path to the configuration file in the experiment that you want to analyze."
    )
    parser.add_argument(
        "-m",
        "--model_pt",
        type=str,
        default= "state_latest.pt",
        help="The name of the checkpoint whose predictions you want to study"
    )
    opts = parser.parse_args()

    # get the model and loader
    checkpoints_dir = pathlib.Path(pathlib.Path(opts.conf_path).parent, "checkpoints")
    model = model_from_run(opts.conf_path, checkpoints_dir, opts.model_pt)
    loader = loader_from_run(opts.conf_path)

    # make predictions and summarize
    y_hat = infer(model, loader)
    histogram(y_hat)
