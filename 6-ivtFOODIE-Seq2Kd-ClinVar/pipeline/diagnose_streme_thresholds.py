#!/usr/bin/env python3
"""Audit the four promoter STREME thresholds without changing existing figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'results/analysis'
OUT = ANALYSIS / 'requested_median_pvalue_figures/threshold_diagnostics'


def compare(d):
    b = d.loc[d.label.eq('Benign'), 'mean'].to_numpy(dtype=float)
    p = d.loc[d.label.eq('Pathogenic'), 'mean'].to_numpy(dtype=float)
    r = stats.mannwhitneyu(p, b, alternative='two-sided', method='auto')
    auc = float(r.statistic / (len(p) * len(b)))
    return dict(benign_n=len(b), pathogenic_n=len(p),
                median_difference=float(np.median(p)-np.median(b)), raw_p=float(r.pvalue),
                auc=auc, cliffs_delta=2*auc-1,
                z_abs=float(stats.norm.isf(r.pvalue/2)),
                no_tie_z=(float(r.statistic)-len(p)*len(b)/2-.5)
                / np.sqrt(len(p)*len(b)*(len(p)+len(b)+1)/12))


def main():
    OUT.mkdir(exist_ok=True)
    frames = {}
    for pct in [1,5,10,30]:
        d = pd.read_parquet(ANALYSIS/f'variant_summaries/promoter.seq2kd_streme.top{pct}.mean.parquet')
        assert not d.variant_id.duplicated().any()
        d = d[d.label.isin(['Benign','Pathogenic'])].copy()
        assert np.isfinite(d['mean']).all()
        frames[pct] = d
    common = set.intersection(*(set(d.variant_id) for d in frames.values()))
    summary, groups = [], []
    originals = pd.read_csv(ANALYSIS/'tables/statistical_tests.csv')
    for pct, d in frames.items():
        row = compare(d)
        ref = originals.loc[(originals.endpoint == f'promoter.seq2kd_streme.top{pct}.mean')
                            & (originals.class_a == 'Benign') & (originals.class_b == 'Pathogenic')
                            & (originals.test == 'Mann-Whitney U')].iloc[0]
        assert np.isclose(row['raw_p'], ref.p, rtol=1e-8, atol=0)
        row.update(top_pct=pct, **{'common_'+k:v for k,v in compare(d[d.variant_id.isin(common)]).items()})
        summary.append(row)
        for label, g in d.groupby('label'):
            q = np.percentile(g['mean'], [10,25,50,75,90])
            groups.append(dict(top_pct=pct, label=label, n=len(g), q10=q[0],q25=q[1],
                               median=q[2],q75=q[3],q90=q[4],iqr=q[3]-q[1],
                               sd=float(g['mean'].std()),median_n_tf=float(g.n_tf.median()),
                               mean_n_tf=float(g.n_tf.mean())))
    summary = pd.DataFrame(summary); groups = pd.DataFrame(groups)
    summary.to_csv(OUT/'threshold_comparisons.csv',index=False)
    groups.to_csv(OUT/'group_distributions.csv',index=False)
    print(summary.to_string(index=False), flush=True)
    print(groups.to_string(index=False), flush=True)
    # Sample eight SNVs per clinical class to independently reconstruct the
    # variant-level effect from feature-level prediction pairs, at all thresholds.
    sample = frames[1].groupby('label',group_keys=False).sample(n=8,random_state=261018)
    cols = ['variant_id','protein_id','tf','ref_pkd','alt_pkd','abs_delta','top1','top5','top10','top30']
    pairs = pd.read_parquet(ANALYSIS/'promoter_pair_scores/seq2kd_streme.parquet',columns=cols,
                            filters=[('variant_id','in',sample.variant_id.tolist())])
    assert not pairs.duplicated(['variant_id','protein_id']).any()
    expected = (pairs.alt_pkd-pairs.ref_pkd).abs()
    assert np.allclose(pairs.abs_delta, expected, rtol=0, atol=0)
    for a,b in [(1,5),(5,10),(10,30)]:
        assert ((~pairs[f'top{a}'])|pairs[f'top{b}']).all()
        assert set(frames[a].variant_id) <= set(frames[b].variant_id)
    audits = []
    for pct,d in frames.items():
        t = pairs[pairs[f'top{pct}']].groupby(['variant_id','tf']).abs_delta.max()
        rec = t.groupby('variant_id').agg(['mean','size'])
        ref = d.set_index('variant_id').loc[rec.index]
        err = float(np.max(abs(rec['mean']-ref['mean'])))
        assert np.allclose(rec['mean'],ref['mean'],rtol=1e-6,atol=1e-8)
        assert (rec['size']==ref.n_tf).all()
        audits.append(dict(top_pct=pct, checked_snvs=len(rec), max_abs_mean_difference=err))
    record = dict(unique_snv_all_thresholds=True, all_pvalues_reproduced=True,
                  common_snv_count=len(common), cohorts_nested=True,
                  sampled_pair_rows=len(pairs), sampled_variants=len(sample),
                  sampled_prediction_abs_delta_exact=True,
                  sampled_threshold_flags_nested=True, sampled_aggregation_checks=audits,
                  scope='Distribution and code audit; sample reconstruction, not full inference revalidation.')
    (OUT/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
