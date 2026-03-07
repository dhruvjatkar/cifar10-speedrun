#!/bin/bash
#SBATCH --job-name=cifar10-075
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=attempts/logs/075_%j.out
#SBATCH --error=attempts/logs/075_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== ATTEMPT 075: Doubled color jitter last 50% - 200 run validation =========="
python attempts/attempt_075_jitter_late_200runs.py
