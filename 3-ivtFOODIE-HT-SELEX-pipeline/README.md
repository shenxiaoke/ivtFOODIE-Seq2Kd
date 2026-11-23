# Reproducible TF-DNA HT-SELEX pipeline

This Snakemake workflow analyzes paired-end TF-binding HT-SELEX data for
NR4A3 and NKX2-6 from raw FASTQ through QC, N40 extraction, exact sequence
counting, enrichment, motif discovery, motif comparison, logos, and a
Seq2Kd-ready sequence-level table.

This is a TF-DNA binding workflow, not an aptamer-selection workflow.

## Additional batch: 20260920 B3 (14 uncharacterized TFs)

A second delivery of fourteen four-cycle TFs is configured independently, so the
original NR4A3/NKX2-6 run and the earlier eight-TF batch are left untouched:

- Config: `config.more_16_uncharacterized_TFs_more.yaml`
- Input: `data_more_16_uncharacterized_TFs_more/` (14 TFs)
- Results/logs: `results_more_16_uncharacterized_TFs_more/`,
  `logs_more_16_uncharacterized_TFs_more/`
- TFs: CENPBD1P, DUX3, FOXD4L1, GTF2I, HMGN3, KMT4, MTERF2, MTERF3, PIN1,
  RHOXF2B, STAB2, TAL2, TBPL2, YBX3
- `motif.sample_seed` is `20260920`.

Select the batch with `--configfile config.more_16_uncharacterized_TFs_more.yaml` when invoking Snakemake. The other bundled YAML files can be selected the same way.

## Dataset-specific interpretation

- `cycle1` has undergone the first selection and is **not** round zero.
- The shared unselected pool is
  `N501-701_S2_L004_R1_001_40nt_sequence.txt` and is named `cycle0`.
- All 615,674 cycle0 rows are retained as observations; duplicate lines
  contribute to counts.
- In selected FASTQs, R1 bases 1-40 and the reverse complement of R2 bases
  1-40 cover the same N40 insert. Base 41 starts the Illumina adapter.
- The experimental sequencing primers are annealing sites and are not present
  in the observed insert. Empty `primer_5` and `primer_3` values are intentional.
- A quality-aware paired consensus is used. Pairs with more than two
  disagreements, or unresolved equal-quality disagreements, are discarded and
  reported in `*.extraction.json`.

## Workflow

```text
raw PE FASTQ ── fastp QC/adapter trimming ── paired N40 consensus
                                               │
shared cycle0 sequence list ───────────────────┤
                                               ▼
                                  external-memory exact counting
                                               ▼
                         raw + depth-normalized enrichment tables
                                      │                    │
                                      ▼                    ▼
                            HTS-IBIS PWM/logo     STREME + TOMTOM
                                      └──────────┬──────────┘
                                                 ▼
                                            HTML report
```

Every computational rule writes a log under `logs/`.

Set the input paths in your selected YAML to the corresponding FASTQ and cycle 0 sequence files. The bundled configurations use paths under `inputs/` that you can populate or replace. Install the external HTS-IBIS executable as documented in [External software](software/README.md).

## 1. Create the independent environment

The environment is intentionally separate from all existing environments:

```bash
mamba env create -f environment.yml
conda activate htselex_env
```

To update an existing installation from the manifest:

```bash
mamba env update -n htselex_env -f environment.yml --prune
```

The environment includes `fastp`, `seqkit`, MEME Suite, Snakemake, scientific
Python packages, OpenJDK, and FASTX-Toolkit (`fastx_trimmer`).
`environment-linux-64.lock.txt` records the exact installed package URLs for
bit-for-bit recreation on Linux x86_64:

```bash
mamba create -n htselex_env --file environment-linux-64.lock.txt
```

## 2. Review or modify `config.yaml`

All experiment paths and tunable parameters are in `config.yaml`; scripts do
not contain experiment-specific paths. Important fields are:

```yaml
extraction:
  random_region_length: 40
  start_1based: 1
  primer_5: ""
  primer_3: ""
motif:
  final_round: cycle4
  background_round: cycle0
```

If a future library truly contains primers inside the reads, do not merely add
primer strings: change the extraction mode or extend
`scripts/02_extract_variable_region.py` and validate it on sampled reads.

## 3. Validate or dry-run

```bash
python scripts/00_validate_config.py --config config.yaml --output /tmp/htselex_config_ok
snakemake -s workflow/Snakefile --configfile config.yaml -n
```

## 4. Run on Slurm

The normal two-command entry point is:

```bash
conda activate htselex_env
sbatch run_pipeline.slurm
```

The job requests 32 CPUs, 64 GB RAM, and three days. Monitor it with:

```bash
squeue -u "$USER"
tail -f logs/slurm/htselex-<JOBID>.out
```

To run interactively inside an adequate allocation:

```bash
snakemake -s workflow/Snakefile --configfile config.yaml --cores 32 \
  --resources mem_mb=64000 --rerun-incomplete --printshellcmds
```

Snakemake resumes completed outputs; `--rerun-incomplete` safely rebuilds
outputs interrupted before atomic completion.

## 5. Results

- `results/clean_fastq/<TF>/`: clean paired FASTQs plus fastp HTML/JSON QC.
- `results/sequences/<TF>/`: N40 sequences, FASTA, FASTQ, extraction statistics.
- `results/counts/<TF>/<round>.counts.tsv.gz`: sorted per-round exact counts.
- `results/counts/<TF>/sequence_counts.tsv.gz`: union of all unique sequences,
  with `cycle0_count` through `cycle4_count`.
- `results/enrichment/<TF>/sequence_enrichment.tsv.gz`: Seq2Kd-ready table.
- `results/motifs/<TF>/hts_ibis/`: HTS-IBIS PWM, MEME format, and logo.
- `results/motifs/<TF>/streme/`: STREME motifs and report.
- `results/motifs/<TF>/tomtom/`: HTS-IBIS versus STREME comparison.
- `results/report/index.html`: combined analysis report.

Large sequence/count tables are gzip-compressed TSV files. Inspect them with:

```bash
zcat results/enrichment/NR4A3/sequence_enrichment.tsv.gz | head
```

### Enrichment definitions

For comparison of target round `r` to source round `b`, with pseudocount
`alpha=1`:

```text
raw = log2((count_r + alpha) / (count_b + alpha))
```

The normalized result compares pseudocount-adjusted frequencies, correcting for
different library depths:

```text
normalized = log2(
  ((count_r + alpha) / (total_r + alpha)) /
  ((count_b + alpha) / (total_b + alpha))
)
```

The table includes every selected cycle versus cycle0 and, by default,
adjacent comparisons (`cycle2` versus `cycle1`, etc.). Raw frequencies and all
round counts are retained.

## Motif sampling and reproducibility

STREME uses at most 100,000 read instances per round. It reports both the
configured final-round versus cycle0 comparison and all adjacent comparisons
(`cycle1_vs_cycle0` through `cycle4_vs_cycle3`). Each adjacent comparison saves
the complete STREME report plus a standalone PNG of its first-ranked motif under
`results/motifs/<TF>/streme_pairwise/`. HTS-IBIS uses at most 1,000,000 per round
across cycle0-cycle4. Sampling is uniform over read instances (so abundance is
preserved), without replacement, and seeded by `motif.sample_seed`. Limits are
configurable and sampling metadata is saved.

The fixed HTS-IBIS PWM is also scanned against every sampled cycle with FIMO
on both strands at the common threshold configured by
`motif.hts_ibis_scan_pvalue` (default `1e-4`). The abundance-weighted fraction
of motif-positive reads, Wilson 95% confidence interval, fold change versus
cycle0, and fold change versus the preceding cycle are written to
`results/motifs/<TF>/hts_ibis/prevalence_by_cycle.tsv`, with a companion PNG.

## HTS-IBIS and BEEML-SELEX

HTS-IBIS is installed under `software/HTS-IBIS/` from the authors' official
Java distribution. Installation provenance and the BEEML/BEESEM compatibility
assessment are documented in `software/README.md`.

The original BEEML-SELEX is not installed because no maintained reproducible
distribution was found and it assumes a known binding-site position. BEESEM is
the more appropriate successor for an unknown site within N40, but its public
implementation requires Python 2.7 plus obsolete dependencies. HTS-IBIS,
STREME, and the complete sequence-level enrichment output are the supported
alternatives in this reproducible Python 3.11 workflow.
