#!/bin/bash
#SBATCH --job-name=cifar10-087-088
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=attempts/logs/087_088_%j.out
#SBATCH --error=attempts/logs/087_088_%j.err

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

echo "========== BASELINE (50 runs) =========="
python attempts/baseline_50runs.py

echo ""
echo "========== ATTEMPT 087: Hybrid Muon 3 NS iters + row preconditioner (50 runs) =========="
python attempts/attempt_087_hybrid_muon_3ns.py

echo ""
echo "========== ATTEMPT 088: Hybrid Muon 2 NS iters + row preconditioner (50 runs) =========="
python attempts/attempt_088_hybrid_muon_2ns.py
