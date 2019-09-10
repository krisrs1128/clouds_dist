# clouds_dist


## Dataset
* The dataset consists of 3100 training samples compressed in npz files 
* Each training sample has the following attributes
    * img (reflectance image): (3, 256, 256) 2d with 3 channels represent the reflectance ranges 0-1
    * Lat, Long: 2d tensors (256, 256) each, zipped with the reflectance images tensor.
    * U, V : 3d tensors (10, 256, 256) represent wind component for 10 different levels.
    * RH: 3d tensor (10, 256, 256) represents relative-humidity for 10 different levels ranges 0-1.
    * Scattering_angle: 2d tensor (256, 256).
    * TS: 2d tensor (256, 256) represents surface temperature.
* The histogram of some of the meteorological vars is shown in figure 1
shows differences in scale  

![alt text](https://github.com/krisrs1128/clouds_dist/raw/master/output/histogram.png "histogram of 5 meteorological vars ")
    

## Training script 
To run the training script, should be able do to something like
```
module load singularity
singularity shell --bind /scratch/sankarak/data/clouds/:/data,/home/sankarak/clouds_dist/:~/ /scratch/sankarak/images/clouds.img
> python3 train.py
```

### Train.py args

`src/train.py` expects these flags:
* `--message | -m "this is a message"`, in the spirit of a commit message, will be added to the comet experiment: `exp.log_parameter("__message", opts.message)`
* `--conf_name | -c largeLRs` the name of the training procedure configuration `json` file. Argument to `conf_name` is the name of the file to be found in `config/`
* `--output_dir | -o $SCRATCH/clouds/runs` where to store the procedure's output: images, comet files, checkpoints conf file etc.
* `--offline | -f` default is a standard comet.ml experiment. on beluga, such experiments are not possible because compute nodes don't have an internet access so use this flag to dump the experiment locally in the model's `output_path`


## Comet.ml

In order to use comet.ml, do add a `.comet.config` in the root of the repo on your machine/cluster:

```
[comet]
api_key=YOUR-API-KEY
workspace=YOUR-WORKSPACE
project_name=THE-PROJECT
```

[WARNING: PROCEDURE DOES NOT WORK FOR NOW, WORK IN PROGRESS]

**Compute nodes don't have internet access!** So to bypass this, install `proxychains` (see hereafter) and then ssh to a login node and the compute node will be able to uplaod to comet! (Based on Joseph's post [Dealing with internet restricted compute nodes in a cluster](http://josephpcohen.com/w/dealing-with-internet-restricted-compute-nodes-in-a-cluster/))

```
ssh -N -D 9050 beluga1 & proxychains4 -q python train.py
```

#### How to set up proxychains

```
$ git clone git@github.com:rofl0r/proxychains-ng.git
$ cd proxychains-ng
$ mkdir ~/.local # don't do this if it already exists
$ ./configure --prefix=$HOME/.local
$ make & make install
$ make install-config
$ export PROXYCHAINS_CONF_FILE=$HOME/.local/etc/proxychains.conf # add this to your ~/.bash_profile
$ proxychains4 ping google.com # should work now
```

## Default conf file

**Remember:** update this section when new arguments are added to the possible configrations

config/defaults.json:

```
{
    "model": {
        "n_blocks": 5,
        "filter_factors": null,
        "kernel_size": 3,
        "dropout": 0.75,
        "Cin": 42,
        "Cout": 3,
        "disc_size": 64
    },
    "train": {
        "n_epochs": 100,
        "lr_d": 0.01,
        "lr_g": 0.0005,
        "lambda_gan": 0.01,
        "lambda_L1": 1,
        "batch_size": 32,
        "n_epoch_regress": 100,
        "n_epoch_gan": 250,
        "datapath": "/home/sankarak/scratch/data/clouds",
        "n_in_mem": 1,
        "early_break_epoch": 0,
        "load_limit": -1,
        "num_workers": 3,
        "num_D_accumulations": 8
    }
}
```

## Running several jobs

Use `parallel_run.py`:

```
python parallel_run.py -e explore-lr.json
```

This script will execute a `sbatch` job for each element listed in explor-lr.json with default arguments:

* `sbatch` params: 
    ```
    {
        "cpus": 8,
        "mem": 32,
        "runtime": "12:00:00",
        "slurm_out": "$HOME/logs/slurm-%j.out",
        "message": "explore exp run 12h",
        "conf_name": "explore",
        "singularity_path": "/scratch/sankarak/images/clouds.img",
        "offline": True
    }
    ```
* training params: `defaults.json` as above.

The argument passed to `-e` should be in `config/` and should not include it in its name.

* [x] Ok `python parallel_run.py -e explore-lr.json`
* [x] Ok `python parallel_run.py -e explore-lr` (refers to `config/explore-lr.json` but no need to specify `.json`)
* [ ] Not Ok `python parallel_run.py -e config/explore-lr.json`

The dictionnary in `explore.json` contains 2 main fields:

* `experiment`, which defines common parameters for one exploration experiment
  * `name`: mandatory, the experiment's name
  * `exp_dir`: mandatory, where to store the experiment's runs
    * runs will be stored in `exp_dir/run0`, `exp_dir/run1` etc.
  * `repeat`: optional, will repeat `n` times the experiment (`"repeat": 2` means the list will be executed twice, so default is `"repeat": 1`)
* `runs`, which is a `list` of models to train:
  * Has to be a `dict`
  * Has to have keys `"sbatch"` and `"config"` mapping to `dict`s
    * `"config"` Has to have keys `"model"` and  `"train"` mapping to potentially empty dicts
    * Minimal (=default model and training procedure) `run` configuration:
      ```
        {
            "sbatch": {},
            "config": {
                "model": {},
                "train": {}
            }
        }
      ```
 the script will override the above parameters with the ones mentionned in the file. Such a file may look like:

```
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
```

This will run 3 sbatch jobs meaning "keep the default sbatch params, but extend runtime to 24h and vary learning rates". The `sbatch`, `config`, `model` and `train` fields are **mandatory**

## Sampling parameters

In `train.py`, the `sample_param` function allows for sampling a parameter from a configuration file: **any** value in the "config" file / field (basically = sub-values of "train" and "model") can be sampled from a `range`, a `list` or a `uniform` interval:

```
...
"train":{
    ...
    "lr_g": 0.001,
    "lr_d": {
        "sample": "range",
        "from": [0.000001, 0.1, 0.01] # a value will be sampled uniformly from [1.0000e-06, 1.0001e-02, ..., 9.0001e-02]
    },
    "lambda_L1": {
        "sample": "list",
        "from": [0.01, 0.1, 1, 10] # a value will be sampled uniformly from this list
    },
    "lambda_gan": {
        "sample": "uniform",
        "from": [0.001, 1] # a value will be sampled uniformly from the interval [0.001 ... 1]
    }
}
```

Note: if you select to sample from "range", as np.arange is used, "from" MUST be a list, and may contain
    only 1 (=min) 2 (min and max) or 3 (min, max, step) values

