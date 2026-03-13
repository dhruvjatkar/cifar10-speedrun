#!/bin/bash
#SBATCH --job-name=cifar10-141
#SBATCH --partition=short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=attempts/logs/141_%j.out
#SBATCH --error=attempts/logs/141_%j.err

set -euo pipefail

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate /home/d.jatkar/.conda/envs/torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

COOLDOWN_SECONDS=60

echo "========== ATTEMPT 141: Adaptive NS 3/2 + 7.85 epochs (200 runs) =========="
date
nvidia-smi
python attempts/attempt_141_adaptive_ns_moreepochs.py

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"
nvidia-smi

echo ""
echo "========== BASELINE (200 runs) =========="
date
python attempts/baseline_200runs.py
