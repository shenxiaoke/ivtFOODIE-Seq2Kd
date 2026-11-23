# ESM-2 and AlphaMissense

Start after STREME coding predictions have completed. Set `CLINVAR_AM_DIR` to the directory containing `AlphaMissense_hg38.tsv.gz` and the GENCODE v32 GFF3. In the Seq2Kd Python environment, from this directory:

```bash
python prepare_esm.py
python alphamissense.py
python validate_esm.py
CLINVAR_GPUS="0" bash workers.sh
python assemble.py
python association.py
python match_streme_missense.py ../results/evolutionary/tables/association_input.parquet ../results/evolutionary/tables/matched_streme.parquet
python three_model_roc_pr.py ../results/evolutionary/tables/matched_streme.parquet ../results/evolutionary/strict_pathogenic_vs_benign
python three_model_roc_pr.py ../results/evolutionary/tables/matched_streme.parquet ../results/evolutionary/pooled_labels --pooled
```

ESM-2 scores use the negative masked LLR in the actual DNA-binding-domain input. AlphaMissense scores enter only after checking the coordinate, reference and alternate amino acids and transcript/protein mapping. The ROC/PR calculation uses the same missense SNVs for all three models, with paired TF-cluster bootstrap intervals. The primary comparison keeps exact Pathogenic and Benign labels. The pooled comparison preserves the older P/LP versus B/LB grouping and writes to its own directory.
