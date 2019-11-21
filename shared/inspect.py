#!/usr/bin/env python
"""
Utilities to Study Model Results
"""
from src.data import get_loader, get_transforms
from src.stats import get_stats
from src.utils import get_opts
import argparse
import numpy as np
import pathlib
import seaborn as sns
import src.gan as gan
import torch


def infer(model, loader, model_opts, M=1000):
    """
    Predictions on a Subset

    :param model: A torch model nn.Module object, to use for making
      predictions.
    :param samples: A torch dataloader object.
    :param M: The maximum number of batches to go through.
    :return y_hat: A 4D torch tensor, with dimensions corresponding to, sample
      x channel x width x height.
    """
    result = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    M = min(len(loader.dataset) / loader.batch_size, M)

    for m, batch in enumerate(loader):
        if m > M: break
        print(f"Inferring batch {m}/{M}")
        x = get_noisy_input_tensor(batch, model_opts)
        y_hat = model.g(x.to(device))
        result.append(y_hat.detach().cpu())
    return torch.cat(result)


def histogram(x, sample_frac=0.3, out_dir=None):
    """
    Values from Random Tensor Indices

    :param x: A torch tensor or arbitrary dimension
    :param sample_frac: A float specifying what fraction of indices to include
      in the histogram
    :out_dir: The directory to save the figure.
    :return None, but saves figure.png in `out_dir`
    """
    if not out_dir:
        out_dir = pathlib.Path.cwd()

    x = x.numpy().flatten()
    indices = np.random.choice(len(x), int(len(x) * sample_frac))
    sns.distplot(x[indices]).figuresavefig(pathlib.Path(out_dir, "histogram.png"))


def y_scatter(y, y_hat, sample_frac=0.3, out_dir=None):
    """
    Scatterplot of y vs. y_hat

    :param y: Pandas data frame of raw output, as saved by save_iterator
    :param y: Pandas data frame of raw predictions, as saved by save_iterator
    :param sample_frac: Proportion of pixels (across w x h x c) to keep when
      plotting
    :out_dir: The directory to save the outputs to.
    """
    if not out_dir:
        out_dir = pathlib.Path.cwd()

    y, y_hat = y.values.flatten(), y_hat.values.flatten()
    indices = np.random.choice(len(y), int(len(y) * sample_frac))
    p = sns.jointplot(
        y[indices],
        y_hat[indices],
        color="black",
        kind="hex",
        bins=400,
        gridsize=50
    )
    p.set_axis_labels('y', 'y_hat', fontsize=16)
    p.savefig(pathlib.Path(out_dir, "scatterplot.png"))


def save_line(z, f, round_level=4):
    """
    1D Array -> String for line in CSV
    """
    str_fun = lambda z: ",".join(np.round(z, round_level).astype(str))
    f.write(str_fun(z.flatten().numpy()))
    f.write("\n")


def get_noise_tensor(model_opts, shape):
    """Functional version of method in src/train.py"""
    b, h, w = shape[0], shape[2], shape[3]
    Ctot = model_opts.Cin + model_opts.Cnoise
    noise_tensor = torch.FloatTensor(b, Ctot, h, w)
    noise_tensor.uniform_(-1, 1)
    return noise_tensor


def get_noisy_input_tensor(batch, model_opts):
    input_tensor = get_noise_tensor(model_opts, batch["metos"].shape)
    input_tensor[:, : model_opts.Cin, :, :] = batch["metos"]
    return input_tensor


def loader_gen(loader, key="metos"):
    """
    Wrapper for Loaders
    """
    for m, batch in enumerate(loader):
        for i in range(len(batch[key])):
            yield batch[key][i]


def tensor_gen(z):
    """
    Tensor -> Iterator
    """
    for i in range(len(z)):
        yield z[i]


def save_iterator(iterator, out_path="x.csv", crop_ix=None, M=1000):
    """
    Save Iterator to File Incrementally
    """
    with open(out_path, "w") as f:
        for m, sample in enumerate(iterator):
            if m > M: break
            print(f"Extracting batch {m} [at most {M} will be saved]")
            cropped = sample
            if crop_ix:
                cropped = cropped[:, crop_ix[0]:crop_ix[1], crop_ix[0]:crop_ix[1]]
            save_line(cropped, f)


def loader_from_run(opts_path, data_path=None):
    """
    Get Loader (with transforms) from Experiment
    """
    if not data_path:
        opts["data"]["path"] = opts["data"]["original_path"]
    else:
        opts["data"]["path"] = data_path

    print("getting transforms")
    transfs = get_transforms(opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("getting stats")
    stats = get_stats(opts, transfs)
    loader, _ = get_loader(opts, transfs, stats)
    return loader


def model_from_run(opts, checkpoints_dir, model_name):
    """
    Get Model from Experiment
    """
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
        default= "/scratch/sankarak/clouds/regression-run--3/run_0/short-run-cropped.yaml",
        help="The full path to the configuration file in the experiment that you want to analyze."
    )
    parser.add_argument(
        "-m",
        "--model_pt",
        type=str,
        default= "state_latest.pt",
        help="The name of the checkpoint whose predictions you want to study"
    )
    args = parser.parse_args()
    opts = get_opts(args.conf_path)

    # get the model and loader
    checkpoints_dir = pathlib.Path(pathlib.Path(args.conf_path).parent, "checkpoints")
    model = model_from_run(opts, checkpoints_dir, args.model_pt)
    loader = loader_from_run(opts)

    # make predictions and summarize
    y_hat = infer(model, loader, opts["model"])
    save_iterator(tensor_gen(y_hat), "y_hat.csv")
    save_iterator(loader_gen(loader, "real_imgs"), "y.csv")
    save_iterator(loader_gen(loader, "metos"), "x.csv", (50, 60))

    # make some plots
    import pdb
    pdb.set_trace()

    one_row = next(tensor_gen(y_hat)).numpy().flatten()
    usecols = np.random.choice(range(len(one_row)), 2000, replace=False)
    y = pd.read_csv("y.csv", header=None, usecols=usecols, names=range(len(one_row)))
    y_hat = pd.read_csv("y_hat.csv", header=None, usecols=usecols, names=range(len(one_row)))
    y_scatter(y, y_hat)
