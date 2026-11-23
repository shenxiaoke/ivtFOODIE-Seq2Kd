#!/usr/bin/env python3
"""External-memory sequence counting and k-way merging of round count tables."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import os
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path


def open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def temp_for(target: str):
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if target.endswith(".gz") else ".tmp"
    fd, temp = tempfile.mkstemp(prefix=path.stem + ".", suffix=suffix, dir=path.parent)
    os.close(fd)
    return temp


def count_sequences(args):
    temp_output = temp_for(args.output)
    temp_dir = args.temp_dir or str(Path(args.output).parent / "sort_tmp")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    decompress = ["pigz", "-cd", args.input] if args.input.endswith(".gz") else ["cat", args.input]
    sort_cmd = [
        "sort", "--parallel", str(args.threads), "-S", args.memory,
        "-T", temp_dir,
    ]
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    total = unique = 0
    try:
        with open_text(temp_output, "wt") as out:
            out.write("sequence\tcount\n")
            p1 = subprocess.Popen(decompress, stdout=subprocess.PIPE)
            p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, text=True, env=env)
            assert p1.stdout is not None and p2.stdout is not None
            p1.stdout.close()
            previous = None
            count = 0
            for line in p2.stdout:
                sequence = line.rstrip("\n").upper()
                if not sequence:
                    continue
                if previous is None:
                    previous, count = sequence, 1
                elif sequence == previous:
                    count += 1
                else:
                    out.write(f"{previous}\t{count}\n")
                    total += count
                    unique += 1
                    previous, count = sequence, 1
            if previous is not None:
                out.write(f"{previous}\t{count}\n")
                total += count
                unique += 1
            if p2.wait() != 0 or p1.wait() != 0:
                raise RuntimeError("gzip/cat or GNU sort failed")
        os.replace(temp_output, args.output)
    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)
        try:
            Path(temp_dir).rmdir()
        except OSError:
            pass
    summary = {"total_reads": total, "unique_sequences": unique}
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    print(json.dumps(summary))


def count_iterator(path: str, round_name: str):
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != ["sequence", "count"]:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line in handle:
            sequence, count = line.rstrip("\n").split("\t")
            yield sequence, round_name, int(count)


def merge_counts(args):
    names, paths = [], []
    for item in args.round:
        name, path = item.split("=", 1)
        names.append(name)
        paths.append(path)
    if len(names) != len(set(names)):
        raise ValueError("Round names must be unique")
    temp_output = temp_for(args.output)
    totals = {name: 0 for name in names}
    unique_per_round = {name: 0 for name in names}
    union_unique = 0
    try:
        iterators = [count_iterator(path, name) for name, path in zip(names, paths)]
        merged = heapq.merge(*iterators, key=lambda row: row[0])
        with open_text(temp_output, "wt") as out:
            out.write("sequence\t" + "\t".join(f"{name}_count" for name in names) + "\n")
            current = None
            values = None
            for sequence, name, count in merged:
                if sequence != current:
                    if current is not None:
                        out.write(current + "\t" + "\t".join(str(values[n]) for n in names) + "\n")
                        union_unique += 1
                    current = sequence
                    values = {n: 0 for n in names}
                values[name] = count
                totals[name] += count
                unique_per_round[name] += 1
            if current is not None:
                out.write(current + "\t" + "\t".join(str(values[n]) for n in names) + "\n")
                union_unique += 1
        os.replace(temp_output, args.output)
    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)
    summary = {
        "round_order": names,
        "total_reads": totals,
        "unique_sequences_per_round": unique_per_round,
        "union_unique_sequences": union_unique,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count")
    count.add_argument("--input", required=True)
    count.add_argument("--output", required=True)
    count.add_argument("--summary")
    count.add_argument("--threads", type=int, default=4)
    count.add_argument("--memory", default="4G")
    count.add_argument("--temp-dir")
    count.set_defaults(func=count_sequences)
    merge = sub.add_parser("merge")
    merge.add_argument("--round", action="append", required=True, help="NAME=COUNT_TSV[.GZ]")
    merge.add_argument("--output", required=True)
    merge.add_argument("--summary", required=True)
    merge.set_defaults(func=merge_counts)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
