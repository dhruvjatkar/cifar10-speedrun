#!/bin/bash
#SBATCH --job-name=cifar10-031
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=attempts/logs/031_%j.out
#SBATCH --error=attempts/logs/031_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 031: LS 0.12 + Muon momentum 0.68 (50 runs) =========="
python attempts/attempt_031_ls12_mom68.py
