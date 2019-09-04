import json
import copy
from pathlib import Path
import subprocess
import os
import argparse
import re


def get_increasable_name(file_path):
    f = Path(file_path)
    if f.exists():
        name = f.name
        s = list(re.finditer("--\d+\.", name))
        if s:
            s = s[-1]
            d = int(s.group().replace("--", "").replace(".", ""))
            d += 1
            i, j = s.span()
            name = name[:i] + f"--{d}" + name[j - 1 :]
        else:
            name = f.stem + "--1.json"
    return f.parent() / name


def write_conf(run_dir, param):
    """Write config file from params to config/conf_name
    If conf_name exisits, increments a counter in the name:
    explore.json -> explore (1).json -> explore (2).json ...
    """
    cname = param["sbatch"].get("conf_name", "overwritable_conf")
    if not cname.endswith(".json"):
        cname += ".json"

    with open(run_dir / cname, "w") as f:
        json.dump(param["config"], f)
    return run_dir / cname


def env_to_path(path):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable

    """
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


""" default config as ref:
{
    "model": {
        "n_blocks": 5,
        "filter_factors": null,
        "kernel_size": 3,
        "dropout": 0.75,
        "Cin": 42,
        "Cout": 3,
    },
    "train": {
        "n_epochs": 100,
        "lr_d": 0.01,
        "lr_g1": 0.0005,
        "lr_g2": 0.0001,
        "lambda_gan_1": 0.01,
        "lambda_L1_1": 1,
        "lambda_gan_2": 0.03,
        "lamdba_L1_2": 1,
        "batch_size": 32,
        "n_epoch_regress": 100,
        "n_epoch_gan": 250,
        "datapath": "/home/sankarak/scratch/data/clouds",
        "n_in_mem": 1,
        "early_break_epoch": 0,
        "load_limit": -1,
        "num_workers": 3,
    },
}"""

"""Possible explore-lr.json
{
    "experiment":{
        "name": " explore-lr-experiment"
    },
    runs: [
    {
        "sbatch": {
            "runtime": "24:00:00",
            "message": "learning rate exploration",
            "conf_name": "explore-lr"
        },
        "config": {
            "model": {},
            "train": {
                "lr_d": 0.001
            }
        }
    },
    {
        "sbatch": {
            "runtime": "24:00:00",
            "message": "learning rate exploration",
            "conf_name": "explore-lr"
        },
        "config": {
            "model": {},
            "train": {
                "lr_d": 0.0001
            }
        }
    },
    {
        "sbatch": {
            "runtime": "24:00:00",
            "message": "learning rate exploration",
            "conf_name": "explore-lr"
        },
        "config": {
            "model": {},
            "train": {
                "lr_g1": 0.001
            }
        }
    }
]
}

"""

default_sbatch = {
    "cpus": 8,
    "mem": 32,
    "runtime": "12:00:00",
    "slurm_out": "$HOME/logs/slurm-%j.out",
    "message": "explore exp run 12h",
    "conf_name": "explore",
    "singularity_path": "/scratch/sankarak/images/clouds.img",
    "offline": True
}


if __name__ == "__main__":

    EXP_ROOT_DIR = Path(os.environ["SCRATCH"]) / "clouds"
    EXP_ROOT_DIR.mkdir(exist_ok=True)
    EXP_ROOT_DIR = EXP_ROOT_DIR / "experiments"
    EXP_ROOT_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exploration_file",
        type=str,
        default="explore.json",
        help="Where to fint the exploration file",
    )
    opts = parser.parse_args()

    default_json_file = "config/defaults.json"
    with open(default_json_file, "r") as f:
        default_json = json.load(f)

    exploration_file = opts.exploration_file
    if not exploration_file.endswith(".json"):
        exploration_file += ".json"
    with open(f"config/{exploration_file}", "r") as f:
        exploration_params = json.load(f)
        assert isinstance(exploration_params, dict)

    exp_name = exploration_params["experiment"].get("name", "explore-experiment")
    exp_dir = EXP_ROOT_DIR / exp_name
    exp_dir = get_increasable_name(exp_dir)
    exp_dir.mkdir()

    params = []
    for p in exploration_params:
        params.append(
            {
                "sbatch": {**default_sbatch, **p["sbatch"]},
                "config": {
                    "model": {**default_json["model"], **p["config"]["model"]},
                    "train": {**default_json["train"], **p["config"]["train"]},
                },
            }
        )

    for i, param in enumerate(params):
        run_dir = exp_dir / f"run_{i}"
        run_dir.mkdir()
        sbp = param["sbatch"]
        conf_path = write_conf(run_dir, param)  # returns Path() from pathlib
        template = f"""
#!/bin/bash
#SBATCH --account=rpp-bengioy               # Yoshua pays for your job
#SBATCH --cpus-per-task={sbp["cpus"]}       # Ask for 6 CPUs
#SBATCH --gres=gpu:1                        # Ask for 1 GPU
#SBATCH --mem={sbp["mem"]}G                 # Ask for 32 GB of RAM
#SBATCH --time={sbp["runtime"]}             # Run for 12h
#SBATCH -o {env_to_path(sbp["slurm_out"])}  # Write the log in $SCRATCH

module load singularity

echo "Starting job"

$DATADIR=/scratch/sankarak/data/clouds/

singularity shell --nv --bind $HOME/clouds_dist:/home/clouds/,$DATADIR {sbp["singularity_path"]} \\
    cd /home/clouds/ && python3 src/train.py \\
        -m "{sbp["message"]}" \\
        -c "{str(conf_path)} \\
        -o {str(run_dir)} \\
        {"-f" if sbp["offline"] else ""}

"""

        file = run_dir / f"run-{sbp['conf_name']}.sh"
        with file.open("w") as f:
            f.write(template)
        # print(subprocess.check_output(f"sbatch {str(file)}", shell=True))