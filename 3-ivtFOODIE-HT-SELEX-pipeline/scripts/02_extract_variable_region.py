#!/usr/bin/env python3
"""Extract the observed N-region and build a quality-aware paired-end consensus."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

DNA = set("ACGT")
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def open_text(path: str, mode: str = "rt") -> TextIO:
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def atomic_path(target: str) -> tuple[str, str]:
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if target.endswith(".gz") else ".tmp"
    fd, tmp = tempfile.mkstemp(prefix=target_path.stem + ".", suffix=suffix, dir=target_path.parent)
    os.close(fd)
    return tmp, str(target_path)


def fastq_records(handle: TextIO):
    while True:
        header = handle.readline().rstrip("\n")
        if not header:
            return
        seq = handle.readline().rstrip("\n")
        plus = handle.readline().rstrip("\n")
        qual = handle.readline().rstrip("\n")
        if not seq or not plus.startswith("+") or len(seq) != len(qual):
            raise ValueError(f"Malformed FASTQ record near {header!r}")
        yield header, seq.upper(), qual


def read_id(header: str) -> str:
    return header.split()[0].lstrip("@").removesuffix("/1").removesuffix("/2")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_TABLE)[::-1]


def consensus_pair(
    seq1: str,
    qual1: str,
    seq2: str,
    qual2: str,
    start: int,
    length: int,
    max_mismatches: int,
    min_quality_delta: int,
):
    end = start + length
    if len(seq1) < end or len(seq2) < end:
        return None, None, "short"
    s1, q1 = seq1[start:end], qual1[start:end]
    s2, q2 = reverse_complement(seq2[start:end]), qual2[start:end][::-1]
    mismatches = sum(a != b for a, b in zip(s1, s2))
    if mismatches > max_mismatches:
        return None, None, "discordant"

    bases: list[str] = []
    qualities: list[str] = []
    for a, qa, b, qb in zip(s1, q1, s2, q2):
        pa, pb = ord(qa) - 33, ord(qb) - 33
        if a == b and a in DNA:
            bases.append(a)
            qualities.append(chr(min(max(pa, pb), 93) + 33))
        elif a in DNA and b not in DNA:
            bases.append(a)
            qualities.append(qa)
        elif b in DNA and a not in DNA:
            bases.append(b)
            qualities.append(qb)
        elif a in DNA and b in DNA and abs(pa - pb) >= min_quality_delta:
            if pa > pb:
                bases.append(a)
                qualities.append(qa)
            else:
                bases.append(b)
                qualities.append(qb)
        else:
            return None, None, "ambiguous"
    return "".join(bases), "".join(qualities), "exact" if mismatches == 0 else "corrected"


def write_record(seq_out, fasta_out, fastq_out, name: str, seq: str, qual: str):
    seq_out.write(seq + "\n")
    fasta_out.write(f">{name}\n{seq}\n")
    fastq_out.write(f"@{name}\n{seq}\n+\n{qual}\n")


def from_sequence_file(args, outputs):
    seq_out, fasta_out, fastq_out = outputs
    stats = {"input_type": "sequence_file", "total": 0, "retained": 0, "invalid": 0}
    with open_text(args.sequence_file) as source:
        for idx, line in enumerate(source, 1):
            stats["total"] += 1
            seq = line.strip().upper()
            if len(seq) != args.length or not set(seq) <= DNA:
                stats["invalid"] += 1
                continue
            write_record(seq_out, fasta_out, fastq_out, f"cycle0_{idx}", seq, "I" * len(seq))
            stats["retained"] += 1
    return stats


def from_paired_fastq(args, outputs):
    seq_out, fasta_out, fastq_out = outputs
    stats = {
        "input_type": "paired_fastq",
        "total_pairs": 0,
        "retained": 0,
        "exact_pairs": 0,
        "corrected_pairs": 0,
        "short_pairs": 0,
        "discordant_pairs": 0,
        "ambiguous_pairs": 0,
    }
    with open_text(args.r1) as h1, open_text(args.r2) as h2:
        it1, it2 = fastq_records(h1), fastq_records(h2)
        while True:
            rec1 = next(it1, None)
            rec2 = next(it2, None)
            if rec1 is None or rec2 is None:
                if rec1 is not None or rec2 is not None:
                    raise ValueError("R1 and R2 contain different numbers of records")
                break
            stats["total_pairs"] += 1
            h1_name, s1, q1 = rec1
            h2_name, s2, q2 = rec2
            if read_id(h1_name) != read_id(h2_name):
                raise ValueError(f"Out-of-sync pairs: {h1_name!r} and {h2_name!r}")
            seq, qual, status = consensus_pair(
                s1, q1, s2, q2, args.start - 1, args.length,
                args.max_mismatches, args.min_quality_delta,
            )
            if seq is None:
                stats[f"{status}_pairs"] += 1
                continue
            write_record(seq_out, fasta_out, fastq_out, read_id(h1_name), seq, qual)
            stats["retained"] += 1
            stats[f"{status}_pairs"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-file")
    source.add_argument("--r1")
    parser.add_argument("--r2")
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--start", type=int, default=1, help="1-based start in each read")
    parser.add_argument("--max-mismatches", type=int, default=2)
    parser.add_argument("--min-quality-delta", type=int, default=1)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fastq", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()
    if args.r1 and not args.r2:
        parser.error("--r2 is required with --r1")
    if args.start < 1 or args.length < 1:
        parser.error("--start and --length must be positive")

    targets = [args.sequences, args.fasta, args.fastq, args.stats]
    temp_targets = [atomic_path(path) for path in targets]
    try:
        with ExitStack() as stack:
            seq_out = stack.enter_context(open_text(temp_targets[0][0], "wt"))
            fasta_out = stack.enter_context(open_text(temp_targets[1][0], "wt"))
            fastq_out = stack.enter_context(open_text(temp_targets[2][0], "wt"))
            if args.sequence_file:
                stats = from_sequence_file(args, (seq_out, fasta_out, fastq_out))
            else:
                stats = from_paired_fastq(args, (seq_out, fasta_out, fastq_out))
        stats.update({"random_region_length": args.length, "start_1based": args.start})
        with open(temp_targets[3][0], "w") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
        for tmp, target in temp_targets:
            os.replace(tmp, target)
    except Exception:
        for tmp, _ in temp_targets:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        raise
    print(json.dumps(stats, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
