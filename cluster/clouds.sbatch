#!/bin/bash
#SBATCH --job-name=clouds-dist
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=def-bengioy
#SBATCH --mem=32G
#SBATCH --time=3:30:00
#SBATCH --output /home/sankarak/logs/clouds-%j.out
#SBATCH --error /home/sankarak/logs/clouds-%j.err

rsync -avz /scratch/sankarak/data/clouds/imgs/ $SLURM_TMPDIR/imgs/
rsync -avz /scratch/sankarak/data/clouds/metos/ $SLURM_TMPDIR/metos/
module load singularity
singularity shell --nv --bind $SLURM_TMPDIR:/data/,$HOME/clouds_dist:/home/clouds/,$SCRATCH/:$SCRATCH/ /scratch/sankarak/images/clouds.img /home/clouds/cluster/clouds.sh