#!/bin/bash
#SBATCH --job-name=cifar10-007
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=attempts/logs/007_%j.out
#SBATCH --error=attempts/logs/007_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 007: High LR Sweep (8 configs x 50 runs) =========="
python attempts/attempt_007_muonh_high_lr_sweep.py
