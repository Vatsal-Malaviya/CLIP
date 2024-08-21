#!/bin/bash
#SBATCH --job-name=train_clip
#SBATCH --output=outputs/_logs/%x_%j_%A_%a_%u_%N_%t.out
#SBATCH --error=outputs/_logs/%x_%j_%A_%a_%u_%N_%t.err
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00  # Adjust time as per the training requirement

module load mamba/latest
source activate vl

srun python src/train_clip.py training.epochs=20
