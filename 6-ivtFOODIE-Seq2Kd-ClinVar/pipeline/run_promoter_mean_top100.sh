#!/bin/bash
#SBATCH --job-name=clinvar-mean-top100
#SBATCH --partition=vip_gpu_01
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
analysis_python="${CLINVAR_PYTHON:-python}"
export CLINVAR_RUN_DIR="$PWD/results/promoter_mean_top100"
mkdir -p "$CLINVAR_RUN_DIR/logs"
"$analysis_python" -u extend_promoter_mean_top100.py prepare --workers 12
worker_pids=()
IFS=',' read -ra analysis_gpus <<< "${CLINVAR_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [ ${#analysis_gpus[@]} -eq 0 ]; then
  echo 'Set CLINVAR_GPU_IDS or use a Slurm GPU allocation' >&2
  exit 1
fi
for analysis_gpu in "${analysis_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="$analysis_gpu" "$analysis_python" -u inference_fast.py --stage promoter > "$CLINVAR_RUN_DIR/logs/worker.$(hostname).gpu${analysis_gpu}.log" 2>&1 &
  worker_pids+=("$!")
done
worker_status=0
for worker_pid in "${worker_pids[@]}"; do
  wait "$worker_pid" || worker_status=1
done
if [ "$worker_status" -ne 0 ]; then exit "$worker_status"; fi
"$analysis_python" -u extend_promoter_mean_top100.py render
for figure in 02_promoter_streme_jaspar_top100_auc_mean 03_promoter_methods_top100_auc_mean; do
  pdftoppm -f 1 -singlefile -scale-to 2200 -png "results/analysis/requested_median_pvalue_figures/$figure.pdf" "$CLINVAR_RUN_DIR/logs/$figure"
done
