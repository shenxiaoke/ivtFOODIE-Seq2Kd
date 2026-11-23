#!/usr/bin/env python3
"""Calculate raw and library-size-normalized HT-SELEX enrichments."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
from pathlib import Path


def open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--baseline", default="cycle0")
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--adjacent", action="store_true")
    args = parser.parse_args()

    with open(args.summary) as handle:
        metadata = json.load(handle)
    rounds = metadata["round_order"]
    if args.baseline not in rounds:
        raise ValueError(f"Baseline {args.baseline!r} is absent from {rounds}")
    totals = metadata["total_reads"]
    alpha = args.pseudocount
    denominators = {name: totals[name] + alpha for name in rounds}
    comparisons = [(args.baseline, r, "vs_cycle0") for r in rounds if r != args.baseline]
    if args.adjacent:
        comparisons.extend((a, b, f"vs_{a}") for a, b in zip(rounds[1:], rounds[2:]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if args.output.endswith(".gz") else ".tmp"
    fd, temp = tempfile.mkstemp(prefix=output.stem + ".", suffix=suffix, dir=output.parent)
    os.close(fd)
    rows = 0
    try:
        with open_text(args.counts) as inp, open_text(temp, "wt") as out:
            header = inp.readline().rstrip("\n").split("\t")
            expected = ["sequence"] + [f"{r}_count" for r in rounds]
            if header != expected:
                raise ValueError(f"Count table header mismatch: {header} != {expected}")
            columns = expected + [f"{r}_frequency" for r in rounds]
            for left, right, label in comparisons:
                columns.extend([
                    f"{right}_log2_enrichment_{label}_raw",
                    f"{right}_log2_enrichment_{label}_normalized",
                ])
            out.write("\t".join(columns) + "\n")
            for line in inp:
                fields = line.rstrip("\n").split("\t")
                sequence = fields[0]
                counts = {r: int(v) for r, v in zip(rounds, fields[1:])}
                frequencies = {r: counts[r] / totals[r] if totals[r] else float("nan") for r in rounds}
                values = [sequence] + [str(counts[r]) for r in rounds]
                values += [f"{frequencies[r]:.12g}" for r in rounds]
                for left, right, _ in comparisons:
                    raw = math.log2((counts[right] + alpha) / (counts[left] + alpha))
                    normalized = math.log2(
                        ((counts[right] + alpha) / denominators[right]) /
                        ((counts[left] + alpha) / denominators[left])
                    )
                    values.extend([f"{raw:.8g}", f"{normalized:.8g}"])
                out.write("\t".join(values) + "\n")
                rows += 1
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

    result = {
        "rows": rows,
        "baseline": args.baseline,
        "pseudocount": alpha,
        "normalization": "Library-size-normalized pseudocount frequencies: (count + alpha) / (total_reads + alpha)",
        "comparisons": [{"from": a, "to": b, "label": label} for a, b, label in comparisons],
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
