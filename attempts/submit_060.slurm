#!/bin/bash
#SBATCH --job-name=cifar10-060
#SBATCH --partition=gpu,gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/060_%j.out
#SBATCH --error=attempts/logs/060_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 060: Standard CutMix alpha=0.5 prob=30% full training (50 runs) =========="
python attempts/attempt_060_cutmix_a05_p30.py
