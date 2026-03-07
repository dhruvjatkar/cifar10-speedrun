#!/bin/bash
#SBATCH --job-name=cifar10-056
#SBATCH --partition=gpu,gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/056_%j.out
#SBATCH --error=attempts/logs/056_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 056: Fixed 5x5 CutMix, last 50% (50 runs) =========="
python attempts/attempt_056_cutmix_fixed5_late.py
