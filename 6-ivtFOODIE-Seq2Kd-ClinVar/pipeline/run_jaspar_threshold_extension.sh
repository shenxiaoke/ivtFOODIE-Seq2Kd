#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
extension_run="$PWD/results/jaspar_seq2kd_promoter_all_thresholds"
while [ ! -f "$extension_run/qc/inference_signature.json" ]; do sleep 2; done
export CLINVAR_RUN_DIR="$extension_run"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
bash worker_node.sh
unset CLINVAR_RUN_DIR
"${CLINVAR_PYTHON:-python}" -u jaspar_promoter_all_thresholds.py render
