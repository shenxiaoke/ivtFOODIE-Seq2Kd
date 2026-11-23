#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="${CLINVAR_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${CLINVAR_PYTHON:-python}"
stage=${1:-status}
if [ "$#" -gt 0 ]; then shift; fi
case "$stage" in
 prepare) "$PY" initialize.py; "$PY" -u prepare.py;;
 candidates) "$PY" -u candidates.py --workers "${WORKERS:-20}" "$@";;
 tasks) "$PY" build_tasks.py coding; "$PY" build_tasks.py promoter;;
 validate) "$PY" test_pipeline.py; "$PY" -u validate_inference.py; "$PY" -u validate_padding.py;;
 infer-local) bash worker_node.sh;;
 analyze) "$PY" -u analyze.py "$@";;
 status) "$PY" status.py;;
 all) "$PY" initialize.py; "$PY" -u prepare.py; "$PY" -u candidates.py --workers "${WORKERS:-20}"; "$PY" build_tasks.py coding; "$PY" build_tasks.py promoter; "$PY" test_pipeline.py; "$PY" -u validate_inference.py; "$PY" -u validate_padding.py; bash worker_node.sh; "$PY" -u analyze.py "$@";;
 *) echo 'Usage: bash run.sh {prepare|candidates|tasks|validate|infer-local|analyze|status|all}' >&2;exit 2;;
esac
