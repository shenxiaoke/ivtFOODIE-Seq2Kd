# Seq2Kd analysis of ClinVar coding and promoter variants

## Inputs and main analysis

Use the paper's frozen ClinVar 2025-06-23 VCF, indexed hg38 FASTA, GENCODE v48 basic annotation, human TF selections and sequences, and STREME outputs derived from the preceding Seq2Kd motif analysis. The motifs are read from `dna_mutation_sk/Seq2Kd_motif/streme_out_100.part[12]/*/streme.txt` under `CLINVAR_ARCHIVE`. Keep the model source, configuration and v20_class_1 checkpoint together under `CLINVAR_MODEL_SOURCE`. Set `CLINVAR_ESM_CHECKPOINT` to the ESM-2 t33 650M checkpoint. The frozen inputs are inventoried by `pipeline/provenance.py`. Set `CLINVAR_SHARE` to the parent of the directory `clinvar/` containing the VCF, genome, GTF and TF selection, and `CLINVAR_TF_DIR` to the human TF tables. Install the dependencies in `pipeline/requirements.lock.txt` in a CUDA-capable environment.

```bash
bash pipeline/run.sh prepare
bash pipeline/run.sh candidates
bash pipeline/run.sh tasks
bash pipeline/run.sh validate
CLINVAR_GPUS="0" bash pipeline/run.sh infer-local
bash pipeline/run.sh analyze
```

- **Coding:** map the sequence-verified SNVs to the selected TF fragments, annotate the protein change, and score WT/ALT against the same 100 STREME-derived DNA sequences. Summarize the mean absolute change per unique SNV.
- **Promoter:** intersect SNVs with protein-coding TSS ±500 bp, build REF/ALT sequence pairs, select candidates for each TF by the top 5% binding score, and compute Seq2Kd ΔpKd. Combine alternative protein fragments within TF by maximum absolute effect, then average the absolute effect across candidate TFs per unique SNV.
- **Comparison:** figures and statistical tests use exact Pathogenic and Benign annotations. The intermediate SNV table preserves the original labels for optional comparisons. Predictive outputs use pKd = −log10(Kd), so ΔpKd is computed as ALT minus REF.

## Auxiliary article analyses

- **JASPAR motifs:** `pipeline/jaspar_coding.py prepare` creates the independent coding workload and motif inventory using JASPAR 2026. Run workers with `CLINVAR_RUN_DIR` set to the absolute JASPAR result directory and then run the script's `render` stage. The promoter equivalent is `jaspar_seq2kd_promoter.py prepare` and `render`. Its `jaspar_promoter_all_thresholds.py` extension supports the four-threshold JASPAR comparison.
- **JASPAR PWM and ProBound:** run `bash pipeline/run.sh candidates --include-probound` and `bash pipeline/run.sh analyze --include-probound --threshold-sensitivity`, then run `pipeline/jaspar_promoter.py` after JASPAR motif preparation. Set `CLINVAR_PROBOUND` to the ProBound model collection and Java scorer location. The PWM uses absolute log2-odds change and ProBound uses the absolute log10 relative Ka ratio. Both use their native scores.
- **Article AUC figures:** run `plot_requested_median_pvalues.py`, `update_discrimination_tables.py` and `plot_requested_auc_figures.py` after the main and motif/method auxiliary stages have completed in the default `pipeline/results` tree. The plotting scripts validate the original cohorts, statistical tables and source data.
- **ESM-2 and AlphaMissense:** see [the evolutionary workflow](pipeline/evolutionary/README.md). It joins the STREME coding results to ESM-2 and validated AlphaMissense annotations for the same missense variants.
