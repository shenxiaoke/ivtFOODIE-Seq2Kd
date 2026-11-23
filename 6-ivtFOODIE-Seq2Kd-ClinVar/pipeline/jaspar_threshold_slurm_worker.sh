#!/bin/bash
# One inference worker, using only the GPU assigned by Slurm.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CLINVAR_RUN_DIR="$PWD/results/jaspar_seq2kd_promoter_all_thresholds"
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
: "${SLURM_JOB_ID:?Expected a Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Expected a GPU assigned by Slurm}"
"${CLINVAR_PYTHON:-python}" -u inference_fast.py --stage promoter
