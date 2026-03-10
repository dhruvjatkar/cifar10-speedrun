#!/bin/bash
#SBATCH --job-name=cifar10-126
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/126_%j.out
#SBATCH --error=attempts/logs/126_%j.err

set -euo pipefail

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== ATTEMPT 126: Hybrid Neumann+CG compiled resolvent =========="
date
nvidia-smi
python attempts/attempt_126_hybrid_neumann_cg.py --ortho-backend resolvent --benchmark-runs 5
