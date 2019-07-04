# clouds_dist

To run the training script, should be able do to somethign like

```
module load singularity
singularity shell --bind /scratch/sankarak/data/clouds/:/data,/home/sankarak/clouds_dist/:~/ /scratch/sankarak/images/clouds.img
> python3 train.py
```

though right now, it seems to be hanging at some point (no errors, just don't see any messages).
