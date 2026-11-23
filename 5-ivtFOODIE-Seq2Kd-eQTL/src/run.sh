#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec "${SEQ2KD_PYTHON:-python}" pipeline.py "$@"
