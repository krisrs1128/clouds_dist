#!/usr/bin/env python
"""
Upload collection of wandb runs
"""
import argparse
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--runs_dir",
    type=str,
    default=".",
    help="Directory containing all the wandb dry runs to upload",
)
parser.add_argument(
    "-p",
    "--project",
    type=str,
    default="clouds_dist",
    help="Project ID to associate the runs to."
)
opts = parser.parse_args()
p = pathlib.Path(opts.runs_dir) # make this an argument
paths = list(p.rglob('*wandb*'))

for p in paths:
    try:
        subprocess.check_output(f"wandb sync {str(p.parent)} --project={opts.project}", shell=True)
    except:
        print(f"could not upload {p.parent}")
