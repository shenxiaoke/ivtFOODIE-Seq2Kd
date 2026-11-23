#!/usr/bin/env python3
"""Measure the prevalence of one fixed PWM across uniformly sampled rounds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def scan(meme: str, fasta: str, threshold: float, max_stored_scores: int) -> int:
    command = [
        "fimo", "--best-site", "--no-qvalue", "--thresh", str(threshold),
        "--max-stored-scores", str(max_stored_scores),
        "--bfile", "--motif--", meme, fasta,
    ]
    print("Running:", " ".join(command), file=sys.stderr)
    hits = 0
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip() and not line.startswith("#"):
                hits += 1
        stderr_output = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
    if stderr_output:
        sys.stderr.write(stderr_output)
    if "Reached max stored scores" in stderr_output:
        raise RuntimeError(
            "FIMO truncated its results (max stored scores reached), which "
            "under-reports motif prevalence; increase --max-stored-scores."
        )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meme", required=True)
    parser.add_argument(
        "--round", action="append", required=True, metavar="NAME=FASTA=SAMPLING_JSON"
    )
    parser.add_argument("--pvalue", type=float, default=1e-4)
    parser.add_argument(
        "--max-stored-scores",
        type=int,
        default=10_000_000,
        help=(
            "FIMO --max-stored-scores. Must be large enough that FIMO does not "
            "raise its reporting threshold and truncate matches (default: 10000000)."
        ),
    )
    parser.add_argument("--tf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--plot", required=True)
    args = parser.parse_args()

    specifications = []
    for value in args.round:
        fields = value.split("=", 2)
        if len(fields) != 3:
            parser.error(f"Invalid --round value: {value}")
        specifications.append(tuple(fields))

    rows = []
    for round_name, fasta, stats_path in specifications:
        with open(stats_path) as handle:
            stats = json.load(handle)
        total = int(stats["sampled_reads"])
        hits = scan(args.meme, fasta, args.pvalue, args.max_stored_scores)
        fraction = hits / total if total else 0.0
        lower, upper = wilson_interval(hits, total)
        rows.append({
            "round": round_name,
            "sampled_reads": total,
            "motif_positive_reads": hits,
            "motif_positive_fraction": fraction,
            "motif_positive_percent": 100 * fraction,
            "ci95_lower_percent": 100 * lower,
            "ci95_upper_percent": 100 * upper,
        })

    baseline = rows[0]["motif_positive_fraction"]
    for index, row in enumerate(rows):
        previous = rows[index - 1]["motif_positive_fraction"] if index else None
        row["fold_vs_cycle0"] = row["motif_positive_fraction"] / baseline if baseline else None
        row["fold_vs_previous"] = (
            row["motif_positive_fraction"] / previous if previous else None
        )

    fields = list(rows[0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "tf": args.tf,
        "motif": args.meme,
        "scanner": "FIMO --best-site, both strands",
        "pvalue_threshold": args.pvalue,
        "max_stored_scores": args.max_stored_scores,
        "denominator": "uniformly sampled read instances (abundance-weighted)",
        "rows": rows,
    }
    with open(args.summary, "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    labels = [row["round"] for row in rows]
    percentages = [row["motif_positive_percent"] for row in rows]
    lower_error = [p - row["ci95_lower_percent"] for p, row in zip(percentages, rows)]
    upper_error = [row["ci95_upper_percent"] - p for p, row in zip(percentages, rows)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(labels, percentages, yerr=[lower_error, upper_error], marker="o", capsize=3)
    ax.set_ylabel("motif-positive reads (%)")
    ax.set_xlabel("selection cycle")
    ax.set_title(f"{args.tf} HTS-IBIS PWM prevalence (FIMO p < {args.pvalue:g})")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    source_stat = Path(args.meme).stat()
    for output in (args.output, args.summary, args.plot):
        os.utime(output, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))


if __name__ == "__main__":
    main()
