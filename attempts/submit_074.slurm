#!/bin/bash
#SBATCH --job-name=cifar10-074
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/074_%j.out
#SBATCH --error=attempts/logs/074_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 074: Doubled color jitter in last 3 epochs (50 runs) =========="
python attempts/attempt_074_jitter_late_last3ep.py
