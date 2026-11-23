#!/usr/bin/env python3
"""Calculate article AUC, Cliff's delta and average precision from SNV summaries."""
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score
import plot_requested_median_pvalues as plot


def main():
    sources = pd.read_csv(plot.OUT / 'source_statistics.csv').set_index('endpoint')
    groups = {
        '01_coding_auc_cliffs_delta.csv': list(sources.index[sources.index.str.startswith('coding.')]),
        '02_promoter_motif_threshold_auc_cliffs_delta.csv': list(
            sources.index[sources.index.str.startswith('promoter.seq2kd_')]),
        '03_promoter_methods_auc_cliffs_delta.csv': [
            'promoter.seq2kd_streme.top5.mean', 'promoter.probound.top5.mean',
            'promoter.pwm_jaspar.top5.mean',
        ],
    }
    metrics = {}
    for endpoint, source in sources.iterrows():
        path = plot.ROOT / source.source
        if plot.sha256(path) != source.source_sha256:
            raise ValueError(f'SNV summary changed: {path}')
        data = pd.read_parquet(path)
        if data.variant_id.duplicated().any():
            raise ValueError(f'Duplicate variants: {endpoint}')
        benign = data.loc[data.label.eq('Benign'), source.value_column].to_numpy()
        pathogenic = data.loc[data.label.eq('Pathogenic'), source.value_column].to_numpy()
        if (len(benign), len(pathogenic)) != (source.benign_n, source.pathogenic_n):
            raise ValueError(f'Unexpected cohort size: {endpoint}')
        scores = np.concatenate((benign, pathogenic))
        labels = np.repeat([0, 1], [len(benign), len(pathogenic)])
        auc = roc_auc_score(labels, scores)
        u = mannwhitneyu(pathogenic, benign).statistic
        delta = 2 * u / (len(benign) * len(pathogenic)) - 1
        np.testing.assert_allclose(delta, 2 * auc - 1, atol=1e-12)
        metrics[endpoint] = dict(
            endpoint=endpoint, benign_n=len(benign), pathogenic_n=len(pathogenic),
            auc_pathogenic_higher=auc, cliffs_delta_pathogenic_minus_benign=delta,
            auprc_average_precision=average_precision_score(labels, scores),
            pathogenic_prevalence_pr_baseline=float(labels.mean()))
    for file_name, endpoints in groups.items():
        pd.DataFrame([metrics[endpoint] for endpoint in endpoints]).to_csv(plot.OUT / file_name, index=False)
    (plot.OUT / 'discrimination_metrics_validation.json').write_text(json.dumps({
        'positive_class': 'Pathogenic', 'negative_class': 'Benign',
        'source_hashes_and_unique_snv_counts_verified': True, 'distinct_endpoints': len(metrics),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
