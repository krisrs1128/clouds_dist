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

though right now, it seems to be hanging at some point (no errors, just don't see any messages).

## Comet.ml

In order to use comet.ml, do add a `.comet.config` in the root of the repo on your machine/cluster:

```
[comet]
api_key=YOUR-API-KEY
workspace=YOUR-WORKSPACE
project_name=THE-PROJECT
```

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

For each dictionnary listed in `explore.json` the script will override the above parameters with the ones mentionned in the file. Such a file may look like:

```
[
    {
        "sbatch": {
            "runtime": "24:00:00",
            "conf_name":"lr_d_001"
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
            "conf_name": "lr_d_0001"
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
            "conf_name": "lr_g1_001"
        },
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
