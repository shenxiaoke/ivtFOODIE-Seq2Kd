import gzip
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def test_extract_count_merge_enrichment_and_sampling(tmp_path):
    sequence = "ACGT" * 10
    reverse_complement = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    adapter1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    adapter2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
    r1, r2 = tmp_path / "r1.fastq.gz", tmp_path / "r2.fastq.gz"
    with gzip.open(r1, "wt") as h1, gzip.open(r2, "wt") as h2:
        h1.write(f"@read1 1:N:0:1\n{sequence + adapter1}\n+\n{'I' * (40 + len(adapter1))}\n")
        h2.write(f"@read1 2:N:0:1\n{reverse_complement + adapter2}\n+\n{'I' * (40 + len(adapter2))}\n")
    selected = tmp_path / "cycle1.txt.gz"
    run(
        SCRIPTS / "02_extract_variable_region.py", "--r1", r1, "--r2", r2,
        "--length", "40", "--sequences", selected,
        "--fasta", tmp_path / "cycle1.fa.gz", "--fastq", tmp_path / "cycle1.fq.gz",
        "--stats", tmp_path / "cycle1.json",
    )
    with gzip.open(selected, "rt") as handle:
        assert handle.read().strip() == sequence
    assert json.loads((tmp_path / "cycle1.json").read_text())["retained"] == 1

    cycle0 = tmp_path / "cycle0.txt"
    cycle0.write_text(f"{sequence}\n{sequence}\n{'A' * 40}\n")
    cycle2 = tmp_path / "cycle2.txt"
    cycle2.write_text(f"{'A' * 40}\n{'A' * 40}\n{'A' * 40}\n")
    count_paths = []
    for name, source in (("cycle0", cycle0), ("cycle1", selected), ("cycle2", cycle2)):
        count_path = tmp_path / f"{name}.counts.tsv.gz"
        run(SCRIPTS / "03_count_sequences.py", "count", "--input", source,
            "--output", count_path, "--threads", "1", "--memory", "10M")
        count_paths.append(count_path)
    merged, summary = tmp_path / "sequence_counts.tsv.gz", tmp_path / "summary.json"
    merge_args = [SCRIPTS / "03_count_sequences.py", "merge"]
    for name, path in zip(("cycle0", "cycle1", "cycle2"), count_paths):
        merge_args += ["--round", f"{name}={path}"]
    run(*merge_args, "--output", merged, "--summary", summary)
    assert json.loads(summary.read_text())["total_reads"] == {"cycle0": 3, "cycle1": 1, "cycle2": 3}

    enrichment = tmp_path / "enrichment.tsv.gz"
    run(
        SCRIPTS / "04_calculate_enrichment.py", "--counts", merged, "--summary", summary,
        "--output", enrichment, "--output-summary", tmp_path / "enrichment.json",
        "--baseline", "cycle0", "--pseudocount", "1", "--adjacent",
    )
    with gzip.open(enrichment, "rt") as handle:
        columns = handle.readline().rstrip("\n").split("\t")
    assert len(columns) == len(set(columns))
    assert "cycle2_log2_enrichment_vs_cycle1_normalized" in columns

    run(
        SCRIPTS / "05_prepare_motif_input.py", "--counts", merged, "--round", "cycle0",
        "--max-sequences", "2", "--seed", "7", "--fasta", tmp_path / "motif.fa",
        "--fastq", tmp_path / "motif.fq", "--stats", tmp_path / "motif.json",
    )
    assert json.loads((tmp_path / "motif.json").read_text())["sampled_reads"] == 2


def test_report_generation(tmp_path):
    cfg = {
        "project": {"name": "test", "description": "test report"},
        "results_dir": "results",
        "samples": {
            "TFX": {
                "baseline": {"name": "cycle0"},
                "rounds": {"cycle1": {}},
            }
        },
        "motif": {"final_round": "cycle1"},
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    results = tmp_path / "results"
    for directory in (
        results / "sequences" / "TFX", results / "counts" / "TFX",
        results / "enrichment" / "TFX", results / "motifs" / "TFX" / "hts_ibis",
        results / "report",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for round_name in ("cycle0", "cycle1"):
        (results / "sequences" / "TFX" / f"{round_name}.extraction.json").write_text(
            json.dumps({"total": 2, "retained": 2})
        )
    (results / "counts" / "TFX" / "sequence_counts.summary.json").write_text(json.dumps({
        "union_unique_sequences": 1,
        "unique_sequences_per_round": {"cycle0": 1, "cycle1": 1},
    }))
    with gzip.open(results / "enrichment" / "TFX" / "sequence_enrichment.tsv.gz", "wt") as handle:
        handle.write("sequence\tcycle1_count\tcycle1_log2_enrichment_vs_cycle0_normalized\n")
        handle.write(f"{'A' * 40}\t2\t1.5\n")
    (results / "report" / "software_versions.txt").write_text("python: test\n")
    output = results / "report" / "index.html"
    run(SCRIPTS / "07_generate_report.py", "--config", config, "--root", tmp_path, "--output", output)
    assert "TFX" in output.read_text()
