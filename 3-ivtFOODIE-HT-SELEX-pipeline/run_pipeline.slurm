#!/usr/bin/env bash
#SBATCH --job-name=htselex
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate htselex_env

snakemake \
  --snakefile workflow/Snakefile \
  --configfile config.yaml \
  --cores "${SLURM_CPUS_PER_TASK:-32}" \
  --resources mem_mb=64000 \
  --rerun-incomplete \
  --printshellcmds \
  --latency-wait 60
