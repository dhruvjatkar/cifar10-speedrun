#!/bin/bash
#SBATCH --job-name=cifar10-020-200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=attempts/logs/020_200_%j.out
#SBATCH --error=attempts/logs/020_200_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== ATTEMPT 020 VALIDATION: LS 0.12 (200 runs) =========="
python attempts/attempt_020_ls12_200runs.py
