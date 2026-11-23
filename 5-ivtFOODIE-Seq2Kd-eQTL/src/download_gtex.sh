#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 GTEX_DIR SUSIE_ARCHIVE_URL_OR_PATH SIGNIFICANT_PAIRS_ARCHIVE_URL_OR_PATH" >&2
    exit 2
fi
mkdir -p -- "$1"
dest=$(cd -- "$1" && pwd)
for spec in "GTEx_v10_SuSiE_eQTL|$2" "GTEx_Analysis_v10_eQTL|$3"; do
    name=${spec%%|*}
    source=${spec#*|}
    archive="$dest/$name.tar"
    if [[ "$source" == https://* ]]; then
        curl --fail --location --retry 3 --output "$archive.partial" "$source"
        mv -- "$archive.partial" "$archive"
    else
        archive="$source"
    fi
    mkdir -p -- "$dest/$name"
    tar -xf "$archive" -C "$dest/$name" --no-same-owner --no-same-permissions
done
