#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
results=${CLINVAR_RUN_DIR:-../results}
export CLINVAR_RUN_DIR="$(realpath -m "$results")"
mkdir -p "$CLINVAR_RUN_DIR/evolutionary/logs"
pids=()
read -r -a gpus <<< "${CLINVAR_GPUS:-0}"
for gpu in "${gpus[@]}"; do
 CUDA_VISIBLE_DEVICES=$gpu "${CLINVAR_PYTHON:-python}" -u esm_worker.py > "$CLINVAR_RUN_DIR/evolutionary/logs/esm.gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
