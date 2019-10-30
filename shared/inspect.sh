#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32
#SBATCH --time=01:20:00
#SBATCH -o /home/sankarak/logs/inspect-%j.out
#SBATCH -e /home/sankarak/logs/inspect-%j.err

cd $HOME/clouds_dist
module load singularity/3.4

singularity exec --nv --bind /scratch/sankarak/data/clouds,/scratch/sankarak/data/clouds,/scratch/sankarak/clouds \
            /scratch/sankarak/images/clouds.img \
            python3 -m shared.inspect -m state_latest.pt
