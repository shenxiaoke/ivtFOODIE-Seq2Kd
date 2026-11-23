#!/usr/bin/env python3
"""Fail early on missing inputs or inconsistent HT-SELEX configuration."""

import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    errors = []
    length = cfg["extraction"].get("random_region_length")
    if not isinstance(length, int) or length < 1:
        errors.append("extraction.random_region_length must be a positive integer")
    for tf, sample in cfg.get("samples", {}).items():
        baseline = Path(sample["baseline"]["sequence_file"])
        if not baseline.is_file():
            errors.append(f"{tf}: missing baseline sequence file: {baseline}")
        for round_name, files in sample.get("rounds", {}).items():
            for mate in ("r1", "r2"):
                path = Path(files[mate])
                if not path.is_file():
                    errors.append(f"{tf}/{round_name}: missing {mate}: {path}")
    jar = Path(cfg["software"]["hts_ibis_jar"])
    if not jar.is_file():
        errors.append(f"HTS-IBIS JAR missing: {jar}")
    if errors:
        raise SystemExit("Configuration validation failed:\n- " + "\n- ".join(errors))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("configuration and input paths validated\n")


if __name__ == "__main__":
    main()
