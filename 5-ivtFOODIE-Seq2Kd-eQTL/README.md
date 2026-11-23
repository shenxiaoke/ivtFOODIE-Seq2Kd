# Seq2Kd association with GTEx eQTL

## GTEx top 5%

1. Download `GTEx_v10_SuSiE_eQTL.tar` and `GTEx_Analysis_v10_eQTL.tar` under GTEx Analysis V10 in the [GTEx Portal](https://gtexportal.org/home/datasets/). Run `bash src/download_gtex.sh inputs/gtex-v10 SUSIE_ARCHIVE PAIRS_ARCHIVE` with the downloaded paths or their HTTPS download URLs.
2. Configure the hg38 FASTA and its existing index, the Seq2Kd v20_class_1 model source/config/checkpoint, human TF proteins and ESM embeddings. The keys in `src/config.json` show the required names. Set absolute paths or run the Python commands from this directory. The eQTL PWM comparison uses the paper's H14CORE primary matrices and the UniProt-to-gene metadata at `source_root/important_data/promoterAI/human_tf_add_GO_term.tsv`. Set the R environment with the article's `org.Hs.eg.db` snapshot.
3. Run the GTEx workflow with the Python environment for Seq2Kd (CUDA, PyTorch, ruamel.yaml, numpy, pandas, scipy, pysam, pyarrow and matplotlib).

```bash
python src/pipeline.py run --config src/config.json --output results/main
```

The preparation stage requires PIP > 0.9 in any SuSiE record for the variant, then groups significant pairs by variant and gene. It keeps pairs supported by at least two tissues with mean TSS distance within ±500 bp, restricts to SNVs, verifies REF against hg38 and constructs matched 40-nt windows. The scoring stage generates REF/ALT predictions in FP32 with a fixed protein crop shared across alleles. For the binding threshold, pool REF and ALT scores per TF across unique variants, then retain variants with either score in the top 5%. OLS fits change in affinity score against the mean slope for each SNP–gene observation. GO enrichment uses the tested and annotated TFs as background. Results are in `results/main/analysis`, with the top-5% GO figure in `results/main/figures`.

The original article's historical prediction has a different run lineage. The `--historical-scores` switch is available for an audit in a separate output directory. Document any remaining differences in model input, crop or fragment coverage before interpreting a rerun as numerically identical to that prediction.

## Auxiliary workflows

- **Threshold sensitivity:** run the same command with `--threshold-sensitivity --output results/threshold-sensitivity` to evaluate 1/5/10/30%.
- **MPRA:** run with `--datasets mpra --output results/mpra`. Set the MPRA assay table and indexed hg19 reference. The preparation stage removes identical assay records, averages within cells, then gives each cell equal weight in the aggregate. The seeded 10,000-SNV sampling record and sequence QC are written with the prepared inputs.
- **Joint article figure:** use `--datasets eqtl mpra --threshold-sensitivity --output results/eqtl-mpra`. The `go_enrichment_all_p` figure compares the two data sources and the two scoring methods across the four thresholds.
- **TF overexpression:** follow [the overexpression analysis](overexpression/README.md) for the exploratory reference-set enrichment and GO overlap figures.
