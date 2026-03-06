#!/bin/bash
#SBATCH --job-name=cifar10-047
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=attempts/logs/047_%j.out
#SBATCH --error=attempts/logs/047_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (200 runs) =========="
python cifar10_speedrun.py

echo ""
echo "========== ATTEMPT 047: LS 0.12 (200 runs) =========="
python attempts/attempt_047_ls12_200runs_v2.py
