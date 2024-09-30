#!/bin/bash

#SBATCH -J astral
#SBATCH -o logs/sbatch_out/%x.%A_%a.%N.out
#SBATCH -e logs/sbatch_out/%x.%A_%a.%N.gerr
#SBATCH -D ./
#SBATCH --get-user-env

#SBATCH --partition=exbio-cpu          # compms-cpu-small | shared-gpu | exbio-gpu
#SBATCH --nodes=1
##SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --tasks-per-node=1

#SBATCH --export=NONE
#SBATCH --time=96:00:00
##SBATCH --array=1-100%3


source ~/miniconda3/etc/profile.d/conda.sh
conda activate dataset

python -u create_full_dataset.py &> $1.log
