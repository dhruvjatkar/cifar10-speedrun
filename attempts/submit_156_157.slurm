#!/bin/bash
#SBATCH --job-name=cifar10-156-157
#SBATCH --partition=short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=attempts/logs/156_157_%j.out
#SBATCH --error=attempts/logs/156_157_%j.err

set -euo pipefail

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate /home/d.jatkar/.conda/envs/torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

COOLDOWN_SECONDS=60

echo "========== ATTEMPT 156: Lean Muon (200 runs) =========="
date
nvidia-smi
python attempts/attempt_156_lean_muon.py

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"
nvidia-smi

echo ""
echo "========== ATTEMPT 157: 2 NS iterations (200 runs) =========="
date
python attempts/attempt_157_ns2iter.py

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"
nvidia-smi

echo ""
echo "========== BASELINE (200 runs) =========="
date
python attempts/baseline_200runs.py
