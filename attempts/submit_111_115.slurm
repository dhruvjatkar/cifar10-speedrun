#!/bin/bash
#SBATCH --job-name=cifar10-111-115
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=attempts/logs/111_115_%j.out
#SBATCH --error=attempts/logs/111_115_%j.err

set -euo pipefail

module load miniconda3/25.9.1
eval "$(conda shell.bash hook)"
conda activate torch241_bench
cd /projects/Sontag_Lab_Storage/cifar10-speedrun

COOLDOWN_SECONDS=60
RUNS=5

echo "========== ATTEMPT 111: Direct solve + 4 terms (${RUNS} runs) =========="
date
nvidia-smi
python attempts/attempt_111_resolvent_4terms.py --benchmark-runs ${RUNS}

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"

echo "========== ATTEMPT 112: Direct solve + 6 terms (${RUNS} runs) =========="
date
python attempts/attempt_112_resolvent_6terms.py --benchmark-runs ${RUNS}

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"

echo "========== ATTEMPT 113: Neumann series (${RUNS} runs) =========="
date
python attempts/attempt_113_resolvent_neumann.py --benchmark-runs ${RUNS}

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"

echo "========== ATTEMPT 114: AOL + 4 terms (${RUNS} runs) =========="
date
python attempts/attempt_114_resolvent_aol.py --benchmark-runs ${RUNS}

echo ""
echo "========== GPU COOLDOWN (${COOLDOWN_SECONDS}s) =========="
date
sleep "${COOLDOWN_SECONDS}"

echo "========== ATTEMPT 115: Full stack Neumann+AOL+FP16+compile (${RUNS} runs) =========="
date
python attempts/attempt_115_resolvent_fast.py --benchmark-runs ${RUNS}

echo ""
echo "========== ALL DONE =========="
date
