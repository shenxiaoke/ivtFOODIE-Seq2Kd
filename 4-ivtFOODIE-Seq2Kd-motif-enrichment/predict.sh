#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: bash predict.sh MODEL_DIR CHECKPOINT EMBEDDINGS INPUT OUTPUT" >&2
    exit 2
fi

model_dir=$(realpath "$1")
checkpoint=$(realpath "$2")
embeddings=$(realpath "$3")
input=$(realpath "$4")
output=$(realpath "$(dirname "$5")")/$(basename "$5")
cd "$model_dir"
torchrun --standalone --nnodes=1 --nproc-per-node=1 inference_tffilter.py \
    --model-path "$checkpoint" --config-path deepproteindna_gen_test4.yaml \
    --output-dir "$output" --input-path "$input" --embeddings-path "$embeddings"
