#!/usr/bin/env python
import wandb
from addict import Dict
import pandas as pd
import argparse
from pathlib import Path


def flatten_config(config):
    config.train.init_keys = "-".join(config.train.init_keys)

    # merge in main components
    result = Dict(config.data)
    result.update(config.model)
    result.update(config.train)

    # parts that don't live in main keys
    result.__message = config.__message
    result.d_num_trainable_params = config.d_num_trainable_params
    result.g_num_trainable_params = config.g_num_trainable_params
    result.train_samples = config.train_samples
    result.transforms = config.transforms
    return result


def configs_df(runs):
    configs = []
    for run in runs:
        config = flatten_config(Dict(run.config))
        config["id"] = run.id
        configs.append(config)

    configs = pd.DataFrame([dict(s) for s in configs])
    return configs.set_index("id").reset_index()


def metrics_df(runs):
    metrics = []
    for run in runs:
        metrics_set = run.history()
        metrics_set["id"] = run.id
        metrics.append(metrics_set)

    metrics = pd.concat(metrics, axis=0, sort=False)
    return metrics.set_index("id").reset_index()


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, default="mustafa", help="Login name of wandb user",)
    parser.add_argument("-p", "--project", type=str, default="clouds_dist_lr", help="Set of runs to download data from",)
    parser.add_argument("-o", "--output", type=str, default=".", help="Directory to write output to",)
    return parser.parse_args()


if __name__ == '__main__':
    opts = args()
    api = wandb.Api()
    runs = api.runs(f"{opts.username}/{opts.project}")
    configs_df(runs).to_csv(
        Path(opts.output, f"config-{opts.project}.csv"),
        index=False
    )
    metrics_df(runs).to_csv(
        Path(opts.output, f"metrics-{opts.project}.csv"),
        index=False
    )
