#!/bin/bash
#SBATCH --job-name=cifar10-013
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/013_%j.out
#SBATCH --error=attempts/logs/013_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 013: CANS NS coefficients + TTA 0.15 (50 runs) =========="
python attempts/attempt_013_cans_tta15.py
