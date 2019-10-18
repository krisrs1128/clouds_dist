import copy
from pathlib import Path
import subprocess
import os
import argparse
import re
import yaml


def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def write_hash(run_dir):
    run_dir = Path(run_dir)
    with Path(run_dir / "hash.txt").open("w") as f:
        f.write(get_git_revision_hash())


def get_template(param, conf_path, run_dir, name):

    zip_command = ""
    original_path = Path(param["config"]["data"]["original_path"]).resolve()
    zip_name = original_path.name + ".zip"
    zip_path = str(original_path / zip_name)
    if not Path(zip_path).exists():
        zip_command = f"""
cd {str(original_path)}
zip -r {zip_name} imgs metos > /dev/null/
"""

    cp_command = f"""
cp {zip_path} $SLURM_TMPDIR
"""

    unzip_command = f"""
cd $SLURM_TMPDIR
unzip {zip_name} > /dev/null
"""

    sbp = param["sbatch"]

    if name == "victor_mila":
        return f"""#!/bin/bash
#SBATCH --cpus-per-task={sbp.get("cpu", 8)}       # Ask for 6 CPUs
#SBATCH --gres={sbp.get("gpu", "gpu:titanxp:1")}                        # Ask for 1 GPU
#SBATCH --mem=32G                 # Ask for 32 GB of RAM
#SBATCH --time={sbp.get("runtime", "24:00:00")}
#SBATCH -o {str(run_dir)}/slurm-%j.out  # Write the log in $SCRATCH

{zip_command}

{cp_command}

{unzip_command}

cd /network/home/schmidtv/clouds_dist

echo "Starting job"

source /network/home/schmidtv/anaconda3/bin/activate

conda activate cyclePT

python -m src.train \\
                -m "{sbp['message']}" \\
                -c "{str(conf_path)}" \\
                -o "{str(run_dir)}" \\
                {"-n" if sbp["no_comet"] else "-f" if sbp["offline"] else ""}

echo 'done'
"""
    elif name == "mustafa_beluga":
        return f"""#!/bin/bash
#SBATCH --account=rpp-bengioy               # Yoshua pays for your job
#SBATCH --cpus-per-task={sbp["cpus"]}       # Ask for 6 CPUs
#SBATCH --gres=gpu:1                        # Ask for 1 GPU
#SBATCH --mem={sbp["mem"]}G                 # Ask for 32 GB of RAM
#SBATCH --time={sbp.get("runtime", "24:00:00")}
#SBATCH -o {env_to_path(sbp["slurm_out"])}  # Write the log in $SCRATCH

{zip_command}

{cp_command}

{unzip_command}

module load singularity

echo "Starting job"

singularity exec --nv --bind {param["config"]["data"]["path"]},{str(run_dir)}\\
        {","+param["config"]["data"]["preprocessed_data_path"] if param["config"]["data"]["preprocessed_data_path"] else "" } \\
        {sbp["singularity_path"]}\\
        python3 -m src.train \\
        -m "{sbp["message"]}" \\
        -c "{str(conf_path)}"\\
        -o "{str(run_dir)}" \\
        {"-n" if sbp["no_comet"] else "-f" if sbp["offline"] else ""}
"""
    else:
        raise ValueError("No template name provided ; try ... -t mustafa_beluga")


def get_increasable_name(file_path):
    f = Path(file_path)
    while f.exists():
        name = f.name
        s = list(re.finditer(r"--\d+", name))
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
    explore.yaml -> explore (1).yaml -> explore (2).yaml ...
    """
    cname = param["sbatch"].get("conf_name", "overwritable_conf")
    if not cname.endswith(".yaml"):
        cname += ".yaml"

    with open(run_dir / cname, "w") as f:
        yaml.dump(param["config"], f, default_flow_style=False)
    return run_dir / cname


default_sbatch = {
    "cpus": 8,
    "mem": 32,
    "runtime": "12:00:00",
    "slurm_out": "$HOME/logs/clouds-job-%j.out",
    "message": "explore exp run 12h",
    "conf_name": "explore",
    "singularity_path": "/scratch/sankarak/images/clouds.img",
    "offline": True,
    "no_comet": False,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exploration_file",
        type=str,
        default="explore.yaml",
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
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="create files but not run sbatch to test",
    )

    opts = parser.parse_args()

    # -----------------------------------------

    default_yaml_file = "shared/defaults.yaml"
    with open(default_yaml_file, "r") as f:
        default_yaml = yaml.safe_load(f)

    exploration_file = opts.exploration_file
    if not Path(exploration_file).exists():
        if not exploration_file.endswith(".yaml"):
            exploration_file += ".yaml"
        if "config" not in exploration_file:
            exploration_file = "config/" + exploration_file
    with open(exploration_file, "r") as f:
        exploration_params = yaml.safe_load(f)
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
                    "model": {
                        **default_yaml["model"],
                        **(p["config"]["model"] if "model" in p["config"] else {}),
                    },
                    "train": {
                        **default_yaml["train"],
                        **(p["config"]["train"] if "train" in p["config"] else {}),
                    },
                    "data": {
                        **default_yaml["data"],
                        **(p["config"]["data"] if "data" in p["config"] else {}),
                    },
                },
            }
        )

    # -----------------------------------------

    for i, param in enumerate(params):
        run_dir = exp_dir / f"run_{i}"
        run_dir.mkdir()
        sbp = param["sbatch"]

        original_data_path = param["config"]["data"]["path"]
        assert original_data_path, 'no value in param["config"]["data"]["path"]'

        param["config"]["data"]["path"] = "$SLURM_TMPDIR"
        param["config"]["data"]["original_path"] = original_data_path

        conf_path = write_conf(run_dir, param)  # returns Path() from pathlib
        write_hash(run_dir)

        template = get_template(param, conf_path, run_dir, opts.template_name)

        file = run_dir / f"run-{sbp['conf_name']}.sh"
        with file.open("w") as f:
            f.write(template)
        if not opts.test_mode:
            print(subprocess.check_output(f"sbatch {str(file)}", shell=True))
        print("In", str(run_dir), "\n")
