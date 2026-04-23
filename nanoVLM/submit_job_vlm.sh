#!/bin/bash
#SBATCH --job-name=nanoVLM
#SBATCH --time=3:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --partition=l40s
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=nanoVLM_%j.out
#SBATCH --error=nanoVLM_%j.err

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

cd /home/doupeux/com-304-FM-project-2026/nanoVLM || exit 1

export OMP_NUM_THREADS=1

# Hugging Face caches on scratch to avoid filling /home
export HF_HOME=/scratch/$USER/hf_home
export HF_DATASETS_CACHE=/scratch/$USER/hf_datasets
export TRANSFORMERS_CACHE=/scratch/$USER/hf_home/transformers
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/hf_home/hub

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# W&B
export WANDB__SERVICE_WAIT=300

# Optional: only if needed
# export WANDB_DIR=/scratch/$USER/wandb
# mkdir -p "$WANDB_DIR"

torchrun --nproc_per_node=2 train.py