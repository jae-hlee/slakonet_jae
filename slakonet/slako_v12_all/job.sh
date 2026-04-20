#!/bin/bash
#SBATCH --partition=h100
#SBATCH --exclude=h06
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=slako12
#SBATCH --output=slakonet_%A_%a.out
#SBATCH --error=slakonet_%A_%a.err
#SBATCH --array=0-19%10

source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate slakonet
cd /weka/scratch/kchoudh2/jlee859/work/slako/slako_v12
python jslako_v12.py
