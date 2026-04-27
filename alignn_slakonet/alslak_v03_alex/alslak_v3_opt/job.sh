#!/bin/bash
#SBATCH --partition=h100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=alignnv5
#SBATCH --output=alignn_%j.out
#SBATCH --error=alignn_%j.err

source /home/jlee859/scratchkchoudh2/jlee859/miniconda3/etc/profile.d/conda.sh
conda activate bmat
cd /home/jlee859/scratchkchoudh2/jlee859/work/alignn_v5
python predict_alignn.py
