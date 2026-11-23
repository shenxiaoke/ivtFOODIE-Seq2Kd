#!/usr/bin/env python3
"""Deterministically downsample read instances from a sequence/count table."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path


def open_text(path, mode="rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fastq", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    count_column = f"{args.round}_count"
    total = 0
    with open_text(args.counts) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if count_column in header:
            idx = header.index(count_column)
        elif header == ["sequence", "count"]:
            idx = 1
        else:
            raise ValueError(f"No {count_column!r} or standard 'count' column in {args.counts}")
        for line in handle:
            total += int(line.rstrip("\n").split("\t")[idx])
    sample_size = min(total, args.max_sequences)
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(range(total), sample_size)) if sample_size < total else None

    Path(args.fasta).parent.mkdir(parents=True, exist_ok=True)
    cursor = emitted = cumulative = 0
    with open_text(args.counts) as handle, open(args.fasta, "w") as fasta, open(args.fastq, "w") as fastq:
        header = handle.readline().rstrip("\n").split("\t")
        idx = header.index(count_column) if count_column in header else header.index("count")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            sequence, count = fields[0], int(fields[idx])
            if count == 0:
                continue
            if selected is None:
                copies = count
            else:
                copies = 0
                end = cumulative + count
                while cursor < sample_size and selected[cursor] < end:
                    copies += 1
                    cursor += 1
                cumulative = end
            for _ in range(copies):
                emitted += 1
                name = f"{args.round}_{emitted}"
                fasta.write(f">{name}\n{sequence}\n")
                fastq.write(f"@{name}\n{sequence}\n+\n{'I' * len(sequence)}\n")
    stats = {
        "round": args.round,
        "population_reads": total,
        "sampled_reads": emitted,
        "sample_seed": args.seed,
        "sampling": "uniform_without_replacement_over_read_instances",
    }
    with open(args.stats, "w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
