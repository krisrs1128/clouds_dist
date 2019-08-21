# clouds_dist

To run the training script, should be able do to something like

```
module load singularity
singularity shell --bind /scratch/sankarak/data/clouds/:/data,/home/sankarak/clouds_dist/:~/ /scratch/sankarak/images/clouds.img
> python3 train.py
```

though right now, it seems to be hanging at some point (no errors, just don't see any messages).


In order to use comet.ml, do add a `.comet.config` in the root of the repo on your machine/cluster:

```
[comet]
api_key=YOUR-API-KEY
workspace=YOUR-WORKSPACE
project_name=THE-PROJECT
```

config/defaults.json:

```
{
    "model": {
        "n_blocks": 5,
        "filter_factors": null,
        "kernel_size": 3,
        "dropout": 0.75,
        "Cin": 42,
        "Cout": 3
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
        "datapath": "/home/vsch/scratch/data/clouds",
        "n_in_mem": 1,
        "early_break_epoch": 0,
        "load_limit": -1,
        "num_workers": 3
    }
}
```

## Running several jobs

Use `parallel_run.py`:

```
python parallel_run.py -e explore-lr.json
```

This script will execute a `sbatch` job for each element listed in explor-lr.json with default arguments:

* sbatch params: 
  * ```
{
    "cpus": 8,
    "mem": 32,
    "runtime": "12:00:00",
    "slurm_out": "/home/vsch/logs/slurm-%j.out",
    "message": "explore exp run 12h",
    "conf_name": "explore",
}
    ```
* training params: `defaults.json` as above.

For each dictionnary listed in `explore.json` the script will override the above parameters with the ones mentionned in the file. Such a file may look like:

```
[
    {
        "sbatch": {"runtime": "24:00:00"},
        "config": {
            "model": {},
            "train": {
                "lr_d": 0.001
            }
        }
    },
    {
        "sbatch": {"runtime": "24:00:00"},
        "config": {
            "model": {},
            "train": {
                "lr_d": 0.0001
            }
        }
    },
    {
        "sbatch": {"runtime": "24:00:00"},
        "config": {
            "model": {},
            "train": {
                "lr_g1": 0.001
            }
        }
    },
]
```

This will run 3 sbatch jobs meaning "keep the default sbatch params, but extend runtime to 24h and vary learning rates". The `sbatch`, `config`, `model` and `train` fields are **mandatory**