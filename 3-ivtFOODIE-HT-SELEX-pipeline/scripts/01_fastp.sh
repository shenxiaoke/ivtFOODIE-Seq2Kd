#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "Usage: $0 R1 R2 OUT_R1 OUT_R2 HTML JSON THREADS Q UNQUAL_PCT N_LIMIT MIN_LEN ADAPTER_R1,ADAPTER_R2" >&2
  exit 2
fi

r1=$1
r2=$2
out_r1=$3
out_r2=$4
html=$5
json=$6
threads=$7
quality=$8
unqualified=$9
n_limit=${10}
min_len=${11}
IFS=',' read -r adapter_r1 adapter_r2 <<< "${12}"

mkdir -p "$(dirname "$out_r1")" "$(dirname "$html")"

fastp \
  --in1 "$r1" --in2 "$r2" \
  --out1 "$out_r1" --out2 "$out_r2" \
  --html "$html" --json "$json" \
  --thread "$threads" \
  --adapter_sequence "$adapter_r1" \
  --adapter_sequence_r2 "$adapter_r2" \
  --qualified_quality_phred "$quality" \
  --unqualified_percent_limit "$unqualified" \
  --n_base_limit "$n_limit" \
  --length_required "$min_len" \
  --correction --overrepresentation_analysis
