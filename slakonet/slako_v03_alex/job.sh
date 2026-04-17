#!/bin/bash
#SBATCH --partition=h100
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=slakodb
#SBATCH --output=slakonet_%j.out
#SBATCH --error=slakonet_%j.err

source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate slakonet
cd /weka/scratch/kchoudh2/jlee859/work/slako_v3
python jslako.py
