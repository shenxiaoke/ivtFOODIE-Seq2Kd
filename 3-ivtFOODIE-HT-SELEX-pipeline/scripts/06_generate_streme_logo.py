#!/usr/bin/env python3
"""Render the first motif in a STREME text file as a PNG logo."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd


def first_motif(path: str) -> tuple[str, pd.DataFrame]:
    with open(path) as handle:
        lines = iter(handle)
        for line in lines:
            if not line.startswith("MOTIF "):
                continue
            motif_name = line.rstrip("\n").split(maxsplit=2)[1]
            for line in lines:
                if "letter-probability matrix:" not in line:
                    continue
                match = re.search(r"\bw=\s*(\d+)", line)
                if not match:
                    raise ValueError(f"Could not determine motif width in {path}")
                width = int(match.group(1))
                rows = []
                for _ in range(width):
                    fields = next(lines).split()
                    if len(fields) != 4:
                        raise ValueError(f"Malformed probability row in {path}")
                    rows.append([float(value) for value in fields])
                frame = pd.DataFrame(rows, columns=list("ACGT"))
                frame = frame.div(frame.sum(axis=1), axis=0)
                return motif_name, frame
    raise ValueError(f"No motif was reported by STREME in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    motif_name, frame = first_motif(args.input)
    information = logomaker.transform_matrix(
        frame, from_type="probability", to_type="information"
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(frame) * 0.45), 2.8))
    logomaker.Logo(information, ax=ax)
    ax.set_ylabel("bits")
    ax.set_xlabel("position")
    ax.set_title(f"{args.title} — {motif_name}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
