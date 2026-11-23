#!/usr/bin/env python3
"""Draw STREME three-model ROC and PR curves on strict ClinVar labels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def draw(input_path, output_dir, pooled=False):
    columns = {
        "Seq2Kd": "mean_abs_delta",
        "ESM-2 DBD": "negative_dbd_LLR",
        "AlphaMissense": "am_score_validated",
    }
    data = pd.read_parquet(input_path)
    benign = ["Benign", "Likely_benign"] if pooled else ["Benign"]
    pathogenic = ["Pathogenic", "Likely_pathogenic"] if pooled else ["Pathogenic"]
    data = data.loc[data.source.eq("streme") & data.label.isin(benign + pathogenic)].copy()
    grouping = "P/LP vs B/LB" if pooled else "Pathogenic vs Benign"
    if data.variant_id.duplicated().any() or data.tf.isna().any():
        raise ValueError("Expected one scored row with a TF per matched variant")
    scores = data[list(columns.values())].to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("The three models must have finite scores for the same variants")
    y = data.label.isin(pathogenic).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Both strict ClinVar classes must be present")

    tfs, cluster = np.unique(data.tf, return_inverse=True)
    weights = np.random.default_rng(42).multinomial(
        len(tfs), np.full(len(tfs), 1 / len(tfs)), size=2000
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(10, 4.4))
    metrics = []
    for (model, col), color, line in zip(columns.items(), ["#356A8A", "#BC8725", "#AB6286"], ["-", "--", "-."]):
        score = data[col].to_numpy()
        auc = roc_auc_score(y, score)
        ap = average_precision_score(y, score)
        boot = []
        for tf_weights in weights:
            row_weights = tf_weights[cluster]
            if row_weights[y].sum() and row_weights[~y].sum():
                boot.append((roc_auc_score(y, score, sample_weight=row_weights),
                             average_precision_score(y, score, sample_weight=row_weights)))
        (auc_low, ap_low), (auc_high, ap_high) = np.quantile(boot, [.025, .975], axis=0)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc.plot(fpr, tpr, color=color, ls=line, label=f"{model}: AUC {auc:.3f} [{auc_low:.3f}, {auc_high:.3f}]")
        pr.plot(recall, precision, color=color, ls=line, label=f"{model}: AP {ap:.3f} [{ap_low:.3f}, {ap_high:.3f}]")
        metrics.append({"model": model, "pathogenic": int(y.sum()), "benign": int((~y).sum()),
                        "tfs": len(tfs), "roc_auc": auc, "auc_ci_low": auc_low, "auc_ci_high": auc_high,
                        "average_precision": ap, "ap_ci_low": ap_low, "ap_ci_high": ap_high,
                        "pr_baseline": y.mean(), "bootstrap_samples": len(boot)})
    roc.plot([0, 1], [0, 1], color="#777777", ls=":")
    pr.axhline(y.mean(), color="#777777", ls=":")
    roc.set(xlabel="False positive rate", ylabel="True positive rate")
    pr.set(xlabel="Recall", ylabel="Precision")
    for ax, location in [(roc, "lower right"), (pr, "lower left")]:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, frameon=False, loc=location)
    fig.suptitle(f"STREME coding · {grouping}", fontsize=13)
    fig.text(.5, .90, f"{int(y.sum()):,} positive · {int((~y).sum()):,} negative · identical missense SNVs for all three models",
             ha="center", fontsize=9)
    fig.text(.5, .027, f"Paired TF-cluster bootstrap, 95% CIs (2,000 draws).  PR baseline = {y.mean():.3f}.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .88))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "three_model_ROC_PR.streme.png", dpi=200)
    plt.close(fig)
    result = pd.DataFrame(metrics)
    result.to_csv(output_dir / "three_model_ROC_PR.streme.metrics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matched_variants", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pooled", action="store_true", help="Reproduce the earlier P/LP vs B/LB comparison")
    args = parser.parse_args()
    draw(args.matched_variants, args.output_directory, args.pooled)
