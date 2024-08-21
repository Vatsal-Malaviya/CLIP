#!/bin/bash
#SBATCH --job-name=test_clip
#SBATCH --output=outputs/_logs/%x_%j.out
#SBATCH --error=outputs/_logs/%x_%j.err
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --time=02:00:00

module load mamba/latest
source activate vl

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1234 --rdzv_backend=c10d src/train_clip.py
