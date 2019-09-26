import json
import copy
from pathlib import Path
import subprocess
import os
import argparse
import re


def get_template(param, sbp, run_dir, name):
    if name == "victor_mila":
        return f"""#!/bin/bash
#SBATCH --cpus-per-task=8       # Ask for 6 CPUs
#SBATCH --gres=gpu:titanxp:1                        # Ask for 1 GPU
#SBATCH --mem=32G                 # Ask for 32 GB of RAM
#SBATCH --time=24:00:00             # Run for 12h
#SBATCH -o {str(run_dir)}/slurm-%j.out  # Write the log in $SCRATCH

cd /network/home/schmidtv/clouds_dist

echo "Starting job"

source /network/home/schmidtv/anaconda3/bin/activate

conda activate cyclePT

python -m src.train \\
                -m "{sbp['message']}" \\
                -c "{str(conf_path)}" \\
                -o "{str(run_dir)}"
echo 'done'
"""
    elif name == "mustafa_beluga":
        return f"""#!/bin/bash
#SBATCH --cpus-per-task=8       # Ask for 6 CPUs
#SBATCH --gres=gpu:titanxp:1                        # Ask for 1 GPU
#SBATCH --mem=32G                 # Ask for 32 GB of RAM
#SBATCH --time=24:00:00             # Run for 12h
#SBATCH -o {str(run_dir)}/slurm-%j.out  # Write the log in $SCRATCH

cd /network/home/muhammem/clouds_dist

echo "Starting job"

source XXXXXXanaconda3/bin/activate

conda activate XXXXXXenvname

python -m src.train \\
                -m "{sbp['message']}" \\
                -c "{str(conf_path)}" \\
                -o "{str(run_dir)}"
echo 'done'
"""
    else:
        return f"""#!/bin/bash
#SBATCH --account=rpp-bengioy               # Yoshua pays for your job
#SBATCH --cpus-per-task={sbp["cpus"]}       # Ask for 6 CPUs
#SBATCH --gres=gpu:1                        # Ask for 1 GPU
#SBATCH --mem={sbp["mem"]}G                 # Ask for 32 GB of RAM
#SBATCH --time={sbp["runtime"]}             # Run for 12h
#SBATCH -o {env_to_path(sbp["slurm_out"])}  # Write the log in $SCRATCH

module load singularity

echo "Starting job"

singularity exec --nv --bind {param["config"]["train"]["datapath"]},{param["config"]["train"]["preprocessed_data_path"]}{str(exp_dir)} {sbp["singularity_path"]}\\
        python3 src/train.py \\
        -m "{sbp["message"]}" \\
        -c "{str(conf_path)}"\\
        -o "{str(run_dir)}" \\
        {"-f" if sbp["offline"] else ""}
"""



def get_increasable_name(file_path):
    f = Path(file_path)
    while f.exists():
        name = f.name
        s = list(re.finditer("--\d+", name))
        if s:
            s = s[-1]
            d = int(s.group().replace("--", "").replace(".", ""))
            d += 1
            i, j = s.span()
            name = name[:i] + f"--{d}" + name[j:]
        else:
            name = f.stem + "--1" + f.suffix
        f = f.parent / name
    return f


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
        "dropout": 0.25,
        "Cin": 44,
        "Cout": 3,
        "Cnoise": 0
    },
    "train": {
        "batch_size": 32,
        "datapath": "/home/sankarak/scratch/data/clouds",
        "preprocessed_data_path": "/scratch/alghali/data/clouds/",
        "preprocessed_data": true,
        "early_break_epoch": 0,
        "infer_every_steps": 5000,
        "lambda_gan": 0.01,
        "lambda_L": 1,
        "load_limit": -1,
        "lr_d": 0.0002,
        "lr_g": 0.00005,
        "matching_loss": "l2",
        "n_epochs": 100,
        "n_in_mem": 1,
        "num_D_accumulations": 8,
        "num_workers": 3,
        "save_every_steps": 5000,
        "store_images": false,
        "with_stats": "on"
    }
}"""

"""Possible explore-lr.json
{
    "experiment":{
        "name": "explore-lr-experiment",
        "exp_dir": "$SCRATCH/clouds",
        "repeat": 1
    },
    "runs": [
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
                    "lr_d": {
                        "sample": "uniform",
                        "from": [0.00001, 0.01]
                    }
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
                    "lr_g": {
                        "sample": "range",
                        "from": [0.00001, 0.01, 0.001]
                    }
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
    "slurm_out": "$HOME/logs/clouds-job-%j.out",
    "message": "explore exp run 12h",
    "conf_name": "explore",
    "singularity_path": "/scratch/sankarak/images/clouds.img",
    "offline": True,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exploration_file",
        type=str,
        default="explore.json",
        help="Where to find the exploration file",
    )
    parser.add_argument(
        "-d",
        "--exp_dir",
        type=str,
        help="Where to store the experiment, overrides what's in the exp file",
    )
    parser.add_argument(
        "-t",
        "--template_name",
        type=str,
        default="default",
        help="what template to use to write the sbatch files",
    )

    opts = parser.parse_args()

    # -----------------------------------------

    default_json_file = "shared/defaults.json"
    with open(default_json_file, "r") as f:
        default_json = json.load(f)

    exploration_file = opts.exploration_file
    if not exploration_file.endswith(".json"):
        exploration_file += ".json"
    with open(f"config/{exploration_file}", "r") as f:
        exploration_params = json.load(f)
        assert isinstance(exploration_params, dict)

    # -----------------------------------------

    EXP_ROOT_DIR = None
    if "exp_dir" in exploration_params["experiment"]:
        EXP_ROOT_DIR = Path(
            env_to_path(exploration_params["experiment"]["exp_dir"])
        ).resolve()
    if opts.exp_dir:
        EXP_ROOT_DIR = opts.exp_dir
    if EXP_ROOT_DIR is None:
        EXP_ROOT_DIR = Path(os.environ["SCRATCH"]) / "clouds"
        EXP_ROOT_DIR.mkdir(exist_ok=True)
        EXP_ROOT_DIR = EXP_ROOT_DIR / "experiments"

    EXP_ROOT_DIR.mkdir(exist_ok=True)

    exp_name = exploration_params["experiment"].get("name", "explore-experiment")
    exp_dir = EXP_ROOT_DIR / exp_name
    exp_dir = get_increasable_name(exp_dir)
    exp_dir.mkdir()

    # -----------------------------------------

    params = []
    exp_runs = exploration_params["runs"]
    if "repeat" in exploration_params["experiment"]:
        exp_runs *= int(exploration_params["experiment"]["repeat"]) or 1
    for p in exp_runs:
        params.append(
            {
                "sbatch": {**default_sbatch, **p["sbatch"]},
                "config": {
                    "model": {**default_json["model"], **p["config"]["model"]},
                    "train": {**default_json["train"], **p["config"]["train"]},
                },
            }
        )

    # -----------------------------------------

    for i, param in enumerate(params):
        run_dir = exp_dir / f"run_{i}"
        run_dir.mkdir()
        sbp = param["sbatch"]
        conf_path = write_conf(run_dir, param)  # returns Path() from pathlib

        template = get_template(param, sbp, run_dir, opts.template_name)

        file = run_dir / f"run-{sbp['conf_name']}.sh"
        with file.open("w") as f:
            f.write(template)
        print(subprocess.check_output(f"sbatch {str(file)}", shell=True))
        print("In", str(run_dir), "\n")
