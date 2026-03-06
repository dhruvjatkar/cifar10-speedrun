#!/bin/bash
#SBATCH --job-name=cifar10-045
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/045_%j.out
#SBATCH --error=attempts/logs/045_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 045: LS 0.12 + SGD momentum 0.85 (50 runs) =========="
python attempts/attempt_045_ls12_sgdmom85.py
