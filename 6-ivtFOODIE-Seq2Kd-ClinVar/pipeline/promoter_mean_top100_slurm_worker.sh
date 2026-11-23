#!/bin/bash
#SBATCH --job-name=clinvar-top100-worker
#SBATCH --partition=vip_gpu_02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CLINVAR_RUN_DIR="$PWD/results/promoter_mean_top100"
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
while [ ! -f "$CLINVAR_RUN_DIR/tasks/CLOSED" ]; do sleep 10; done
: "${CUDA_VISIBLE_DEVICES:?Expected a Slurm GPU allocation}"
"${CLINVAR_PYTHON:-python}" -u inference_fast.py --stage promoter
