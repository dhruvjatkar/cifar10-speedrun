#!/bin/bash
#SBATCH --job-name=cifar10-069
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/069_%j.out
#SBATCH --error=attempts/logs/069_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 069: MixUp alpha=0.2 first 40% only (mWh-inspired) (50 runs) =========="
python attempts/attempt_069_mixup_early_only.py
