#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 HTS_IBIS_JAR JAVA_MEMORY OUTPUT_PWM ROUND0.fastq [ROUND1.fastq ...]" >&2
  exit 2
fi

jar=$1
memory=$2
output=$3
shift 3

[[ -s "$jar" ]] || { echo "HTS-IBIS JAR not found: $jar" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
java -Xmx"$memory" -jar "$jar" "$output" "$@"
[[ -s "$output" ]] || { echo "HTS-IBIS did not create $output" >&2; exit 1; }
