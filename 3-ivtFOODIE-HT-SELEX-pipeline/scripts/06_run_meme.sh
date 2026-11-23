#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "Usage: $0 POSITIVE.fa CONTROL.fa OUTDIR THREADS MINW MAXW NMOTIFS HTS_MEME TOMTOM_OUT" >&2
  exit 2
fi

positive=$1
control=$2
outdir=$3
threads=$4
minw=$5
maxw=$6
nmotifs=$7
hts_meme=$8
tomtom_out=$9

mkdir -p "$outdir" "$(dirname "$tomtom_out")"
export OMP_NUM_THREADS="$threads"
streme --p "$positive" --n "$control" --dna \
  --oc "$outdir" --minw "$minw" --maxw "$maxw" \
  --nmotifs "$nmotifs" --thresh 0.05

[[ -s "$outdir/streme.txt" ]] || { echo "STREME output missing" >&2; exit 1; }
tomtom -oc "$tomtom_out" -no-ssc -dist pearson -thresh 0.1 "$hts_meme" "$outdir/streme.txt"

# Protect the dependency check from the cluster's cross-node clock skew.
touch -r "$hts_meme" \
  "$outdir/streme.txt" "$outdir/streme.html" "$tomtom_out/tomtom.html"
