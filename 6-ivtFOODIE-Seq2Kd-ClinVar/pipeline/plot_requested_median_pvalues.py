#!/usr/bin/env python3
"""Reproduce supplied results from SNV summaries and export three requested PDFs.

Run in the ClinVar Python environment.
Each paired bar uses the original condition-specific cohort, without matching
cohorts between motif sources. Likely clinical classes are excluded.
"""
from pathlib import Path
import hashlib
import json

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'analysis' / 'requested_median_pvalue_figures'
BLUE, GOLD = '#376895', '#BB8426'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.linewidth': .8,
                     'savefig.facecolor': 'white'})

# endpoint, display label, Benign n, Pathogenic n, supplied rounded raw p
SPECS = [
    ('coding.streme.all.n100', 'STREME', 823, 2585, 6.45e-190),
    ('coding.jaspar.all.n100', 'JASPAR', 234, 1665, 1.19e-66),
    ('promoter.seq2kd_streme.top1.mean', 'STREME\ntop 1%', 8806, 2688, 2.646e-12),
    ('promoter.seq2kd_streme.top5.mean', 'STREME\ntop 5%', 9048, 2736, 1.737e-35),
    ('promoter.seq2kd_streme.top10.mean', 'STREME\ntop 10%', 9049, 2736, 4.363e-49),
    ('promoter.seq2kd_streme.top30.mean', 'STREME\ntop 30%', 9049, 2736, 4.918e-64),
    ('promoter.seq2kd_jaspar.top5.mean', 'JASPAR\ntop 5%', 9049, 2736, 5.075e-23),
    ('promoter.probound.top5.mean', 'ProBound', 9033, 2728, 2.252e-15),
    ('promoter.pwm_jaspar.top5.mean', 'PWM–JASPAR', 9049, 2736, 1.638e-25),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate():
    tests = [RESULTS / 'analysis/tables/statistical_tests.csv',
             RESULTS / 'jaspar_coding/analysis/statistical_tests_unadjusted.csv',
             RESULTS / 'jaspar_seq2kd_promoter/analysis/statistical_tests_unadjusted.csv',
             RESULTS / 'jaspar_promoter/statistical_tests_unadjusted.csv']
    tables = {p: pd.read_csv(p) for p in tests}
    rows = []
    for endpoint, label, nb, np_, expected_p in SPECS:
        path = RESULTS / 'analysis/variant_summaries' / (endpoint + '.parquet')
        if not path.exists():
            path = RESULTS / 'jaspar_coding/analysis/variant_summaries' / (endpoint + '.parquet')
        d = pd.read_parquet(path)
        assert not d.variant_id.duplicated().any(), endpoint
        col = 'value' if endpoint.startswith('coding.') else 'mean'
        b = d.loc[d.label.eq('Benign'), col].to_numpy(dtype=float)
        p = d.loc[d.label.eq('Pathogenic'), col].to_numpy(dtype=float)
        assert np.isfinite(b).all() and np.isfinite(p).all(), endpoint
        assert (len(b), len(p)) == (nb, np_), endpoint
        matches = []
        for tablepath, table in tables.items():
            mask = (table.endpoint.eq(endpoint) & table.test.eq('Mann-Whitney U')
                    & table.class_a.eq('Benign') & table.class_b.eq('Pathogenic'))
            for _, r in table.loc[mask].iterrows():
                matches.append((tablepath, r))
        assert matches, endpoint
        testpath, reference = matches[0]
        raw_p = float(reference.p)
        recomputed_p = float(mannwhitneyu(b, p, alternative='two-sided', method='auto').pvalue)
        assert np.isclose(raw_p, recomputed_p, rtol=1e-8, atol=0), (endpoint, raw_p, recomputed_p)
        assert np.isclose(raw_p, expected_p, rtol=.005, atol=0), endpoint
        assert int(reference.n_a) == nb and int(reference.n_b) == np_, endpoint
        unit = ('|delta PWM log2-odds score|' if '.pwm_' in endpoint else
                '|log10(relative Ka ALT / REF)|' if '.probound.' in endpoint else '|delta pKd|')
        rows.append(dict(endpoint=endpoint, label=label, benign_n=nb, pathogenic_n=np_,
                         benign_median=float(np.median(b)), pathogenic_median=float(np.median(p)),
                         median_difference=float(np.median(p) - np.median(b)), unit=unit,
                         raw_p=raw_p, recomputed_raw_p=recomputed_p,
                         neg_log10_p=float(-np.log10(raw_p)),
                         source=str(path.relative_to(ROOT)), value_column=col,
                         source_sha256=sha256(path), test_source=str(testpath.relative_to(ROOT))))
    return pd.DataFrame(rows), tests


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E4E4E4', linewidth=.65)
    ax.tick_params(axis='x', length=0, pad=10)


def labels_with_n(rows, names=None):
    names = list(rows.label) if names is None else names
    return [f'{name}\nP n={int(r.pathogenic_n):,}\nB n={int(r.benign_n):,}'
            for name, r in zip(names, rows.itertuples())]


def auc_upper(values):
    """Leave label space above the largest AUC with a fixed lower bound of .5."""
    return min(1., .5 + (float(np.max(values)) - .5) * 1.30)


def paired_bars(rows, title, filename, width, use_auc=False):
    fig, ax = plt.subplots(figsize=(width, 5.5))
    right = ax.twinx()
    x = np.arange(len(rows))
    w = .30
    a = ax.bar(x - .18, rows.median_difference, width=w, color=BLUE,
               edgecolor=BLUE, linewidth=.8, zorder=3)
    metric = rows.auc_pathogenic_higher if use_auc else rows.neg_log10_p
    baseline = .5 if use_auc else 0.
    b = right.bar(x + .18, metric - baseline, bottom=baseline, width=w, color='#F7ECD4',
                  edgecolor=GOLD, hatch='///', linewidth=.9, zorder=3)
    ax.set_ylabel(r'Median $|\Delta pK_d|$: Pathogenic $-$ Benign', color=BLUE, labelpad=10)
    right.set_ylabel('AUC' if use_auc else r'$-\log_{10}(p)$', color=GOLD, labelpad=12)
    ax.tick_params(axis='y', colors=BLUE)
    right.tick_params(axis='y', colors=GOLD)
    ax.spines['left'].set_color(BLUE)
    ax.spines['right'].set_visible(False)
    right.spines['right'].set_color(GOLD)
    right.spines['left'].set_visible(False)
    low = min(0., float(rows.median_difference.min()) * 1.27)
    high = max(0., float(rows.median_difference.max()) * 1.27)
    ax.set_ylim(low, high if high > low else low + 1.)
    right.set_ylim((.5, auc_upper(metric)) if use_auc else (0, metric.max() * 1.27))
    ax.set_xlim(-.65, len(rows) - .35)
    ax.set_xticks(x, labels_with_n(rows))
    ax.bar_label(a, labels=[f'{v:.4f}' for v in rows.median_difference], padding=5, color=BLUE, fontsize=10)
    right.bar_label(b, labels=[f'{v:.4f}' if use_auc else f'{v:.2f}' for v in metric], padding=5, color=GOLD, fontsize=10)
    decorate(ax)
    right.grid(False)
    fig.suptitle(title, y=.97, fontsize=14)
    fig.legend(handles=[Patch(facecolor=BLUE, edgecolor=BLUE, label='Median difference (left axis)'),
                        Patch(facecolor='#F7ECD4', edgecolor=GOLD, hatch='///',
                              label='AUC (right axis)' if use_auc else r'$-\log_{10}(p)$ (right axis)')],
               loc='upper center', bbox_to_anchor=(.5, .905), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .025, ('P: Pathogenic; B: Benign. AUC positive class: Pathogenic; AUC axis starts at 0.5.'
                       if use_auc else 'P: Pathogenic; B: Benign. Raw two-sided Mann–Whitney U p values.'),
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.11 if width < 10 else .09, right=.89 if width < 10 else .92,
                        bottom=.27, top=.79)
    fig.savefig(OUT / filename)
    plt.close(fig)


def methods_bars(rows, use_auc=False):
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(rows))
    metric = rows.auc_pathogenic_higher if use_auc else rows.neg_log10_p
    baseline = .5 if use_auc else 0.
    bars = ax.bar(x, metric - baseline, bottom=baseline, width=.48, color='#F7ECD4', edgecolor=GOLD,
                  hatch='///', linewidth=.9, zorder=3)
    ax.set_ylabel('AUC' if use_auc else r'$-\log_{10}(p)$', labelpad=10)
    names = ['Seq2Kd–STREME', 'ProBound', 'PWM–JASPAR']
    quantities = [
        'Predicted binding affinity (pKd)\n' + r'$|\Delta pK_d|$',
        'Relative binding affinity (Ka)\n' + r'$|\log_{10}(K_{a,\mathrm{ALT}}/K_{a,\mathrm{REF}})|$',
        'PWM motif score (log2-odds)\n' + r'$|\Delta\mathrm{motif\ score}|$',
    ]
    labels = [f'{base}\n\n{quantity}\nMedian(P) − Median(B) = {r.median_difference:+.5f}'
              for base, quantity, r in zip(labels_with_n(rows, names), quantities, rows.itertuples())]
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim((.5, auc_upper(metric)) if use_auc else (0, metric.max() * 1.23))
    ax.set_xlim(-.65, len(rows) - .35)
    ax.spines['right'].set_visible(False)
    ax.bar_label(bars, labels=[f'{v:.4f}' if use_auc else f'{v:.2f}' for v in metric], padding=6, fontsize=11)
    decorate(ax)
    fig.suptitle('Promoter variants: methods at top 5% (mean)', y=.96, fontsize=14)
    fig.text(.5, .025, ('P: Pathogenic; B: Benign. AUC positive class: Pathogenic; AUC axis starts at 0.5.\n'
                        if use_auc else 'P: Pathogenic; B: Benign. Raw two-sided Mann–Whitney U p values.\n') +
             'Median differences use method-specific absolute effects (mean across TFs).',
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.09, right=.98, bottom=.39, top=.85)
    fig.savefig(OUT / ('03_promoter_method_auc.pdf' if use_auc else '03_promoter_method_pvalues.pdf'))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tests = calculate()
    rows.to_csv(OUT / 'source_statistics.csv', index=False)
    paired_bars(rows.iloc[:2], 'Coding variants: DNA motif sources (100 DNA)',
                '01_coding_motif_comparison.pdf', 7.8)
    paired_bars(rows.iloc[2:7], 'Promoter variants: motif sources and binding thresholds (mean)',
                '02_promoter_motif_threshold_comparison.pdf', 12.5)
    methods = ['promoter.seq2kd_streme.top5.mean', 'promoter.probound.top5.mean',
               'promoter.pwm_jaspar.top5.mean']
    methods_bars(rows.set_index('endpoint', drop=False).loc[methods].reset_index(drop=True))
    record = dict(script_sha256=sha256(Path(__file__)), pdf_fonttype=mpl.rcParams['pdf.fonttype'],
                  unique_snv_checks=True, finite_values_checks=True,
                  paper_sample_counts_and_pvalues_verified=len(rows),
                  pvalues_recomputed=len(rows),
                  test='two-sided Mann-Whitney U, method=auto; original raw p, no FDR adjustment',
                  statistic='median(Pathogenic variant-level score) - median(Benign variant-level score)',
                  cohorts='Original condition-specific cohorts; Benign and Pathogenic only; no matching across methods.',
                  method_comparison='p value bars; method-specific median differences and predicted quantities below each method',
                  test_table_sha256={str(p.relative_to(ROOT)): sha256(p) for p in tests})
    (OUT / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(rows[['endpoint', 'benign_n', 'pathogenic_n', 'median_difference', 'raw_p', 'neg_log10_p']].to_string(index=False))
    print(f'PDFs saved to {OUT}', flush=True)


if __name__ == '__main__':
    main()
