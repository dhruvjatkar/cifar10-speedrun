#!/bin/bash
#SBATCH --job-name=cifar10-132
#SBATCH --partition=short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/132_%j.out
#SBATCH --error=attempts/logs/132_%j.err

set -euo pipefail

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

COOLDOWN_SECONDS=60

echo "========== ATTEMPT 132: Rewrite PT2.4 (200 runs) =========="
date
nvidia-smi
python attempts/attempt_132_rewrite_pt24.py

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"
nvidia-smi

echo ""
echo "========== BASELINE (200 runs) =========="
date
python attempts/baseline_200runs.py
