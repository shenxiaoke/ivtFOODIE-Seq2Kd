#!/usr/bin/env python3
"""Convert the four-row HTS-IBIS probability matrix to MEME format and PNG."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--meme", required=True)
    parser.add_argument("--logo", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    matrix = {}
    with open(args.input) as handle:
        for line in handle:
            match = re.match(r"^\s*([ACGT]):\s*(.*)$", line)
            if match:
                matrix[match.group(1)] = [float(x) for x in match.group(2).split()]
    if set(matrix) != set("ACGT"):
        raise ValueError(f"Could not parse A/C/G/T rows from {args.input}")
    widths = {len(values) for values in matrix.values()}
    if len(widths) != 1:
        raise ValueError("HTS-IBIS rows have inconsistent widths")
    width = widths.pop()
    frame = pd.DataFrame({base: matrix[base] for base in "ACGT"})
    frame = frame.div(frame.sum(axis=1), axis=0)

    Path(args.meme).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meme, "w") as out:
        out.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        out.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        out.write(f"MOTIF {args.name}\n")
        out.write(f"letter-probability matrix: alength= 4 w= {width}\n")
        for _, row in frame.iterrows():
            out.write(" ".join(f"{row[b]:.8f}" for b in "ACGT") + "\n")

    fig, ax = plt.subplots(figsize=(max(5, width * 0.45), 2.7))
    logomaker.Logo(frame, ax=ax)
    ax.set_ylabel("probability")
    ax.set_xlabel("position")
    ax.set_title(args.name)
    fig.tight_layout()
    fig.savefig(args.logo, dpi=180)
    plt.close(fig)

    # Compute nodes are not perfectly time-synchronized.  Align downstream
    # output mtimes to the input so Snakemake does not mistake clock skew for
    # stale output immediately after a successful conversion.
    source_stat = Path(args.input).stat()
    for output in (args.meme, args.logo):
        os.utime(output, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))


if __name__ == "__main__":
    main()
