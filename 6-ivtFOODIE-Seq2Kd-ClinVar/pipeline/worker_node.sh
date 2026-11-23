#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${CLINVAR_PYTHON:-python}"
RESULT_DIR=${CLINVAR_RUN_DIR:-results}
mkdir -p "$RESULT_DIR/logs"
WORKER_RUN_TAG=${SLURM_JOB_ID:-local.$(date +%s)}
pids=()
read -r -a gpus <<< "${CLINVAR_GPUS:-0}"
for gpu in "${gpus[@]}"; do
 CUDA_VISIBLE_DEVICES=$gpu "$PY" -u inference_fast.py > "$RESULT_DIR/logs/worker.$(hostname).gpu${gpu}.${WORKER_RUN_TAG}.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
