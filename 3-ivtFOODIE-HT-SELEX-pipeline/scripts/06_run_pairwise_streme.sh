#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 POSITIVE.fa CONTROL.fa OUTDIR THREADS MINW MAXW NMOTIFS TITLE" >&2
  exit 2
fi

positive=$1
control=$2
outdir=$3
threads=$4
minw=$5
maxw=$6
nmotifs=$7
title=$8

mkdir -p "$(dirname "$outdir")"
export OMP_NUM_THREADS="$threads"
streme --p "$positive" --n "$control" --dna \
  --oc "$outdir" --minw "$minw" --maxw "$maxw" \
  --nmotifs "$nmotifs" --thresh 0.05

[[ -s "$outdir/streme.txt" ]] || { echo "STREME output missing" >&2; exit 1; }
python scripts/06_generate_streme_logo.py \
  --input "$outdir/streme.txt" \
  --output "$outdir/motif1.logo.png" \
  --title "$title"
