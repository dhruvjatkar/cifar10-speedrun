#!/bin/bash
#SBATCH --job-name=cifar10-046
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/046_%j.out
#SBATCH --error=attempts/logs/046_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 046: LS 0.12 + SGD momentum 0.85 combo (50 runs) =========="
python attempts/attempt_046_ls12_sgdmom85_combo.py
