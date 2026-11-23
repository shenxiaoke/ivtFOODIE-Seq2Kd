"""Sample cCRE windows, prepare Seq2Kd inputs, and call motifs from predictions."""

import argparse
import csv
import heapq
import math
import re
import subprocess
from pathlib import Path


def sample(args):
    import numpy as np
    import pysam

    with args.bed.open() as source:
        regions = [
            (chrom, int(start), int(end))
            for chrom, start, end, *_ in map(str.split, source)
        ]
    total = sum((end - start) // 40 for _, start, end in regions)
    selected = iter(sorted(np.random.RandomState(42).choice(total, 200_000, replace=False)))
    index = next(selected, None)
    offset = 0
    with pysam.FastaFile(str(args.genome)) as genome, args.output.open("w") as out:
        for chrom, start, end in regions:
            count = (end - start) // 40
            while index is not None and index < offset + count:
                left = start + (index - offset) * 40
                dna = genome.fetch(chrom, left, left + 40).upper()
                out.write(f"{chrom}_{left}_{left + 40}\t{dna}\n")
                index = next(selected, None)
            offset += count


def prepare(args):
    with args.windows.open() as source:
        sequences = [line.split()[1] for line in source]
    if len(sequences) != 200_000:
        raise ValueError("Expected exactly 200,000 sampled windows")
    with args.proteins.open(newline="") as source:
        proteins = {row["uniprot"]: row["sequence"] for row in csv.DictReader(source)}
    ids = args.tf_list.read_text().splitlines()
    missing = set(ids) - proteins.keys()
    if missing:
        raise ValueError(f"Missing protein sequences: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=False)
    for start in range(0, len(ids), 50):
        with (args.output / f"batch_{start // 50:03d}.tsv").open("w") as out:
            for tf in ids[start:start + 50]:
                out.writelines(f"{proteins[tf]}\t0\t{dna}\n" for dna in sequences)


def enrich(args):
    predictions = {}
    totals = {}
    for path in args.predictions:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                protein = row["protein_name"]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", protein) or protein in {".", ".."}:
                    raise ValueError(f"Invalid TF identifier: {protein}")
                count = totals.get(protein, 0)
                totals[protein] = count + 1
                if count >= 200_000:
                    continue
                score = float(row["predicted_value"])
                if not math.isfinite(score):
                    raise ValueError(f"Invalid prediction for {protein}")
                dna = row["dna_sequence"].removeprefix("[CLS]")
                entry = (score, -count, dna)
                top = predictions.setdefault(protein, [])
                if len(top) < 100:
                    heapq.heappush(top, entry)
                elif entry > top[0]:
                    heapq.heapreplace(top, entry)
    if not totals or min(totals.values()) < 200_000:
        raise ValueError("Every TF must have at least 200,000 predictions")
    for protein, hits in sorted(predictions.items()):
        fasta = args.output / "fasta_100" / f"{protein}.fasta"
        fasta.parent.mkdir(parents=True, exist_ok=True)
        with fasta.open("w") as out:
            for rank, (_, _, dna) in enumerate(sorted(hits, reverse=True)):
                out.write(f">{protein}_{rank}\n{dna}\n")
        call_streme(fasta, args.output, args.streme)


def call_streme(fasta, output, streme):
    motif = output / "streme_out_100" / fasta.stem
    motif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([streme, "--nmotifs", "1", "--oc", str(motif), "--p", str(fasta)], check=True)


def replay(args):
    fastas = sorted(args.fasta_dir.glob("*.fasta"))
    if not fastas:
        raise ValueError(f"No FASTA files in {args.fasta_dir}")
    for fasta in fastas:
        call_streme(fasta, args.output, args.streme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(required=True)
    for action, paths in (
        (sample, ("bed", "genome", "output")),
        (prepare, ("windows", "proteins", "tf-list", "output")),
        (enrich, ("output",)),
        (replay, ("fasta_dir", "output")),
    ):
        stage = stages.add_parser(action.__name__)
        stage.set_defaults(action=action)
        for path in paths:
            stage.add_argument(path.replace("-", "_"), type=Path)
        if action is enrich:
            stage.add_argument("predictions", nargs="+", type=Path)
        if action in (enrich, replay):
            stage.add_argument("--streme", default="streme")
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
