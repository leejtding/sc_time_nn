#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request hours of runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.

#SBATCH -n 1

# Use more memory (16GB):
#SBATCH --mem=16G

# Send email
#SBATCH --mail-type=end
#SBATCH --mail-user=lee_ding@brown.edu

# Specify a job name:
#SBATCH -J TestRNNLM

# Specify an output file
#SBATCH -o TestRNNLM-%j.out
#SBATCH -e TestRNNLM-%j.out

module load cuda

export PYTHONUNBUFFERED=TRUE
python3 main.py --nhid 300 --nlayers 1 --epochs 15 --dropout 0 --cuda
python3 generate.py