# Article figure entry points

The entries below illustrate how to trace a figure from workflow to source table. The AUC motif and method figures compare Pathogenic and Benign and use one score per SNV.

| Figure example | Stage and source |
| --- | --- |
| `01_coding_motif_comparison_auc.pdf` | STREME and JASPAR Seq2Kd summaries with 100 DNA sequences per variant. |
| `02_promoter_motif_threshold_comparison_auc.pdf` | Mean absolute effects for Seq2Kd–STREME top 1%, 5%, 10%, 30% and Seq2Kd–JASPAR top 5%. |
| `03_promoter_method_auc.pdf` | Top 5% mean absolute effects for Seq2Kd–STREME, JASPAR PWM and ProBound, from their respective per-SNV tables. |
| `three_model_ROC_PR.streme.png` | `evolutionary/match_streme_missense.py` assembles the common cohort. `evolutionary/three_model_roc_pr.py` renders strict or pooled label versions into separate directories and writes a metrics CSV. |

The first three figure commands and the matched-cohort workflow are documented in [README.md](README.md).
