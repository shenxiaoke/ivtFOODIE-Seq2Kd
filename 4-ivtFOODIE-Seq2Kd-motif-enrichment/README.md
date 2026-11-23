# Seq2Kd TF motif enrichment

The workflow samples 200,000 GRCh38 cCRE windows, predicts binding scores with the paper's Seq2Kd v20_class_1 model, then runs MEME Suite STREME 5.5.7 on the 100 highest-scoring predictions per TF. STREME is run with one motif and its default background generation.

## Run from the reference genome

Use the original six-column `GRCh38-cCREs.bed` in its original row order, an indexed `hg38.fa`, and the paper's Seq2Kd model directory, checkpoint, cached human TF embeddings and protein sequence CSV. Use `all_human_tf.junchen20250426.uniprot_seq_data.csv` with columns `uniprot` and `sequence`. The included ID lists were derived from the archived top-100 FASTA filenames (771 main, 878 data2, one shared identifier). Each input shard contains up to 50 proteins. Sampling requires NumPy and pysam. Inference requires the original Seq2Kd Python and CUDA environment with `inference_tffilter.py`, `deepproteindna_gen_test4.yaml`, and the imported model modules in the model directory.

```bash
mkdir -p work/main work/data2
python pipeline.py sample GRCh38-cCREs.bed hg38.fa work/windows.tsv
for batch in main data2; do
    python pipeline.py prepare work/windows.tsv /path/to/human_tf_proteins.csv \
        "${batch}.tf-ids.txt" "work/${batch}/shards"
    for input in work/"$batch"/shards/*.tsv; do
        bash predict.sh /path/to/model /path/to/v20_class_1.pt /path/to/embeddings.pkl \
            "$input" "${input%.tsv}.csv"
    done
    python pipeline.py enrich "work/${batch}" work/"$batch"/shards/*.csv
done
```

For sharded inference, pass the prediction CSVs to `enrich` in their original shard order. Ranking uses descending `predicted_value`, takes the first 200,000 rows per protein before sorting, and uses input order to resolve exact ties. STREME receives the DNA emitted by the historical inference interface after stripping the `[CLS]` prefix. All 164,900 archived FASTA records across the two batches matched 39-nt prefixes of the sampled 40-nt windows. Results are written to `fasta_100/` and `streme_out_100/` under each batch directory.

## Re-run STREME from the archived paper FASTAs

The paper uses both `all_human_tf.20wDNA.v20_class_1.streme` and its nested `all_human_tf.20wDNA.v20_class_1.data2.streme`. Run each archive's `fasta_100/` separately to preserve its TF outputs.

```bash
python pipeline.py replay /path/to/main/fasta_100 work/paper-main
python pipeline.py replay /path/to/data2/fasta_100 work/paper-data2
```

## Sampling provenance

The historical `invivo_foodie_data/sk_process.ipynb` tiles each BED interval into consecutive, non-overlapping 40-nt windows, drops an incomplete last window, and samples 200,000 window indices without replacement with NumPy seed 42. It writes them in increasing source index order. On A100, `sample` reproduced the original `GRCh38-cCREs.seq.20w.txt` byte for byte. The sample SHA-256 is `5cfd1262fbe85b9fd0e7bcfa2999a6f3d15406ef182f9eb61e49ccb2b9137649`.

The BED, genome, Seq2Kd assets, and archived FASTAs are external workflow inputs. Full GPU inference and equality of the new predictions with both historical batches have yet to be verified.
