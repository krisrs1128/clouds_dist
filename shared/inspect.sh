#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:20:00
#SBATCH -o /scratch/sankarak/clouds/

cd $HOME/clouds_dist
module load singularity/3.4

singularity exec --nv --bind /scratch/sankarak/data/clouds,/scratch/sankarak/data/clouds,/scratch/sankarak/clouds \
            /scratch/sankarak/images/clouds.img \
            python3 -m shared.inspect -m state_latest.pt
