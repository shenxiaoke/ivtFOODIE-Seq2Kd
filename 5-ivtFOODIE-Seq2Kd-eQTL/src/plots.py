"""Static, exportable scientific comparison figures."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_figures(out):
    make_go_enrichment_figures(out)
    make_overlap_figures(out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10})
    d = pd.read_csv(out / 'analysis/regressions.csv')
    ds = d[d.group == 'aggregate']
    groups = list(ds.groupby(['dataset', 'method']))
    nrows = max(1, (len(groups) + 1) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.5 * nrows), constrained_layout=True, squeeze=False)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for ax, ((dataset, method), sub) in zip(axes.flat, groups):
        labels = sorted(sub.percent.unique())
        for role, color, offset in [('activator_like', '#bf4968', -.18), ('repressor_like', '#30618a', .18)]:
            counts = [sub[(sub.percent == p) & (sub.role == role)].tf.nunique() for p in labels]
            bars = ax.bar(np.arange(len(labels)) + offset, counts, width=.35, label=role.replace('_', ' '), color=color)
            ax.bar_label(bars, fontsize=8)
        ax.set(xticks=np.arange(len(labels)), xticklabels=[f'{p:g}%' for p in labels], title=f'{dataset} / {method}', ylabel='Unique TFs')
        ax.legend(fontsize=8)
    fig.savefig(dest / 'tf_counts.pdf')
    fig.savefig(dest / 'tf_counts.png', dpi=180)
    plt.close(fig)
    sets = pd.read_csv(out / 'analysis/percent_set_comparisons.csv')
    for dataset, method in ds[['dataset', 'method']].drop_duplicates().itertuples(index=False):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        percentages = sorted(ds.percent.unique())
        for ax, role in zip(axes, ['activator_like', 'repressor_like']):
            matrix = np.eye(len(percentages))
            for r in sets[(sets.dataset == dataset) & (sets.method == method) & (sets.group == 'aggregate') & (sets.role == role)].itertuples():
                i, j = percentages.index(r.p1), percentages.index(r.p2)
                matrix[i, j] = matrix[j, i] = r.jaccard
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
            for i in range(len(percentages)):
                for j in range(len(percentages)):
                    ax.text(j, i, f'{matrix[i,j]:.2f}' if np.isfinite(matrix[i,j]) else 'NA', ha='center', va='center', color='white' if matrix[i,j]>.6 else 'black')
            ax.set(xticks=range(len(percentages)), yticks=range(len(percentages)), xticklabels=[f'{p:g}%' for p in percentages], yticklabels=[f'{p:g}%' for p in percentages], title=role.replace('_', ' '))
        fig.colorbar(im, ax=axes, label='TF set Jaccard')
        fig.suptitle(f'{dataset} / {method}: binding-percent sensitivity')
        fig.savefig(dest / f'{dataset}.{method}.jaccard.pdf')
        fig.savefig(dest / f'{dataset}.{method}.jaccard.png', dpi=180)
        plt.close(fig)
    common = ds.pivot_table(index=['dataset', 'percent', 'tf'], columns='method', values='r_value', aggfunc='mean').dropna()
    datasets = sorted(common.index.get_level_values('dataset').unique())
    if not datasets or not {'seq2kd', 'pwm'}.issubset(common.columns):
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), constrained_layout=True, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        sub = common.loc[dataset]
        for percent in sorted(sub.index.get_level_values('percent').unique()):
            z = sub.loc[percent]
            ax.scatter(z.seq2kd, z.pwm, s=8, alpha=.45, label=f'{percent:g}%')
        ax.axhline(0, color='gray', lw=.7)
        ax.axvline(0, color='gray', lw=.7)
        ax.set(title=dataset, xlabel='Seq2Kd regression r', ylabel='PWM regression r')
        ax.legend(fontsize=8)
    fig.savefig(dest / 'seq2kd_vs_pwm_r.pdf')
    fig.savefig(dest / 'seq2kd_vs_pwm_r.png', dpi=180)
    plt.close(fig)


def make_overlap_figures(out):
    """Compare percent-specific TF sets; empty smaller sets have undefined overlap."""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    pairs = pd.read_csv(out / 'analysis/percent_set_comparisons.csv')
    pairs = pairs[pairs.group == 'aggregate'].copy()
    members = pd.read_csv(out / 'analysis/tf_memberships.csv')
    members = members[members.group == 'aggregate']
    lookup = {key: set(sub.tf) for key, sub in members.groupby(['dataset', 'method', 'role', 'percent'])}
    values = []
    for row in pairs.itertuples():
        a = lookup.get((row.dataset, row.method, row.role, row.p1), set())
        b = lookup.get((row.dataset, row.method, row.role, row.p2), set())
        if (len(a), len(b), len(a & b)) != (row.n1, row.n2, row.intersection):
            raise ValueError('Set comparison counts disagree with TF memberships')
        values.append(len(a & b) / min(len(a), len(b)) if a and b else np.nan)
    pairs['overlap_coefficient'] = values
    pairs.to_csv(dest / 'overlap_coefficient.source.csv', index=False)
    # Combined figures use the same reviewed memberships for both metrics.
    combos = [('eqtl', 'seq2kd'), ('eqtl', 'pwm'), ('mpra', 'seq2kd'), ('mpra', 'pwm')]
    roles = ['activator_like', 'repressor_like']
    percentages = sorted(set(pairs.p1) | set(pairs.p2))
    combined_rows = []
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 11}):
        for metric, title in [('overlap_coefficient', 'Overlap coefficient'), ('jaccard', 'Jaccard index')]:
            fig, axes = plt.subplots(4, 2, figsize=(9, 14), layout='constrained')
            for row_index, (dataset, method) in enumerate(combos):
                for col_index, role in enumerate(roles):
                    ax = axes[row_index, col_index]
                    matrix = np.full((len(percentages), len(percentages)), np.nan)
                    for i, p1 in enumerate(percentages):
                        for j, p2 in enumerate(percentages):
                            a = lookup.get((dataset, method, role, p1), set())
                            b = lookup.get((dataset, method, role, p2), set())
                            denom = min(len(a), len(b)) if metric == 'overlap_coefficient' else len(a | b)
                            v = len(a & b) / denom if denom else np.nan
                            matrix[i, j] = v
                            combined_rows.append({'metric': metric, 'dataset': dataset, 'method': method,
                                                  'role': role, 'p1': p1, 'p2': p2, 'n1': len(a), 'n2': len(b),
                                                  'intersection': len(a & b), 'value': v})
                            if i < j:
                                saved = pairs[(pairs.dataset == dataset) & (pairs.method == method) &
                                              (pairs.role == role) & (pairs.p1 == p1) & (pairs.p2 == p2)]
                                if len(saved) != 1 or not np.isclose(v, saved.iloc[0][metric], equal_nan=True):
                                    raise ValueError(f'{metric} mismatch against saved comparison')
                    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
                    for i in range(len(percentages)):
                        for j in range(len(percentages)):
                            v = matrix[i, j]
                            ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'NA', ha='center', va='center',
                                    color='white' if v > .6 else 'black', fontsize=12)
                    label = f'{"eQTL" if dataset == "eqtl" else "MPRA"} / {"Kd" if method == "seq2kd" else "PWM"}'
                    ax.set(xticks=range(len(percentages)), yticks=range(len(percentages)),
                           xticklabels=[f'{p:g}%' for p in percentages], yticklabels=[f'{p:g}%' for p in percentages],
                           title=f'{label} · {role.replace("_", " ")}', xlabel='Binding threshold p%', ylabel='Binding threshold p%')
            fig.colorbar(im, ax=axes, label=title, shrink=.65, pad=.025)
            fig.suptitle(f'TF set similarity across binding thresholds: {title}', fontsize=16)
            formula = '|A ∩ B| / min(|A|, |B|); empty smaller set = NA' if metric == 'overlap_coefficient' else '|A ∩ B| / |A ∪ B|; empty union = NA'
            fig.supxlabel(f'Aggregate effects · {formula}', fontsize=10)
            fig.savefig(dest / f'all_datasets_methods.{metric}.pdf')
            fig.savefig(dest / f'all_datasets_methods.{metric}.png', dpi=200)
            plt.close(fig)
    pd.DataFrame(combined_rows).to_csv(dest / 'all_datasets_methods.set_similarity.source.csv', index=False)
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10}):
        for (dataset, method), sub in pairs.groupby(['dataset', 'method']):
            percentages = sorted(set(sub.p1) | set(sub.p2))
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), constrained_layout=True)
            for ax, role in zip(axes, ['activator_like', 'repressor_like']):
                matrix = np.full((len(percentages), len(percentages)), np.nan)
                for i, percent in enumerate(percentages):
                    if lookup.get((dataset, method, role, percent)):
                        matrix[i, i] = 1.
                for row in sub[sub.role == role].itertuples():
                    i, j = percentages.index(row.p1), percentages.index(row.p2)
                    matrix[i, j] = matrix[j, i] = row.overlap_coefficient
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
                for i in range(len(percentages)):
                    for j in range(len(percentages)):
                        v = matrix[i, j]
                        ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else 'NA',
                                ha='center', va='center', color='white' if v > .6 else 'black')
                ax.set(xticks=range(len(percentages)), yticks=range(len(percentages)),
                       xticklabels=[f'{p:g}%' for p in percentages],
                       yticklabels=[f'{p:g}%' for p in percentages], title=role.replace('_', ' '))
            fig.colorbar(im, ax=axes, label='Overlap coefficient (0–1)', shrink=.85)
            fig.suptitle(f'{dataset} / {method}: |A ∩ B| / min(|A|, |B|)')
            fig.supxlabel('Binding threshold p%; 1 = smaller set fully contained; empty set = NA', fontsize=9)
            fig.savefig(dest / f'{dataset}.{method}.overlap_coefficient.pdf')
            fig.savefig(dest / f'{dataset}.{method}.overlap_coefficient.png', dpi=200)
            plt.close(fig)
    (dest / 'overlap_coefficient.validation.json').write_text(json.dumps({
        'group': 'aggregate', 'pairs_checked_against_memberships': len(pairs),
        'membership_count_check_passed': True, 'empty_smaller_set': 'NA',
        'formula': '|A intersection B| / min(|A|, |B|)'
    }, indent=2) + '\n')
    print(f'Exported overlap coefficient figures: {len(pairs)} validated pairs', flush=True)


def make_go_enrichment_figures(out, exclusive=False, include_exclusive=True):
    """Plot all dataset x method x p x role x target-GO combinations, aggregate only."""
    import colorsys
    import hashlib
    import itertools
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import hypergeom, fisher_exact

    out = Path(out)
    source = out / 'analysis/go_target_enrichment.csv'
    if not source.exists():
        return
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    d = pd.read_csv(source)
    d = d[d.group == 'aggregate'].copy()
    stem = 'go_enrichment_exclusive_all_p' if exclusive else 'go_enrichment_all_p'
    if exclusive:
        annotations = pd.read_csv(out / 'prepared/human_go_all.csv').dropna().drop_duplicates()
        all_annotated = set(annotations.SYMBOL)
        act = set(annotations.loc[annotations.GOALL == 'GO:0001228', 'SYMBOL'])
        rep = set(annotations.loc[annotations.GOALL == 'GO:0001227', 'SYMBOL'])
        targets = {'GO:0001228': act - rep, 'GO:0001227': rep - act}
        regressions = pd.read_csv(out / 'analysis/regressions.csv')
        backgrounds = {}
        selections = {}
        for key, sub in regressions[regressions.group == 'aggregate'].groupby(['dataset', 'method', 'percent']):
            backgrounds[key] = set(sub.loc[sub.status == 'ok', 'tf']) & all_annotated
            for role in ['activator_like', 'repressor_like']:
                selections[(*key, role)] = set(sub.loc[sub.role == role, 'tf']) & backgrounds[key]
        for ix, row in d.iterrows():
            key = (row.dataset, row.method, row.percent)
            universe = backgrounds[key]
            selected = selections[(*key, row.role)]
            annotated = targets[row.term] & universe
            hits = selected & annotated
            table = [[len(hits), len(selected - annotated)],
                     [len(annotated - selected), len(universe - (selected | annotated))]]
            odds, pval = fisher_exact(table, alternative='greater')
            assert len(universe) == row.n_background and len(selected) == row.n_selected
            assert not (hits & (act & rep))
            d.loc[ix, ['n_annotated', 'n_hits', 'odds_ratio', 'p_value', 'hits']] = [len(annotated), len(hits), odds, pval, ';'.join(sorted(hits))]
        for _, ix in d.groupby(['dataset', 'method', 'group', 'percent']).groups.items():
            vals = d.loc[ix, 'p_value'].to_numpy()
            order = np.argsort(vals)
            adjusted = np.minimum.accumulate((vals[order] * len(vals) / np.arange(1, len(vals) + 1))[::-1])[::-1]
            restored = np.empty(len(vals))
            restored[order] = np.minimum(adjusted, 1)
            d.loc[ix, 'q_value_four_tests'] = restored
        source = out / 'analysis/go_target_enrichment_exclusive.csv'
        d.to_csv(source, index=False)
    datasets = [dataset for dataset in ('eqtl', 'mpra') if dataset in set(d.dataset)]
    methods = ['seq2kd', 'pwm']
    roles = ['activator_like', 'repressor_like']
    terms = ['GO:0001228', 'GO:0001227']
    percentages = sorted(d.percent.unique())
    if not exclusive and datasets == ['eqtl'] and percentages == [5]:
        stem = 'go_enrichment_eqtl_top5'
    keys = ['dataset', 'method', 'percent', 'role', 'term']
    wanted = set(itertools.product(datasets, methods, percentages, roles, terms))
    actual = set(d[keys].itertuples(index=False, name=None))
    if d.duplicated(keys).any() or actual != wanted:
        raise ValueError('Missing, unexpected, or duplicate aggregate GO enrichment combinations')
    pvalues = d.p_value.to_numpy(dtype=float)
    if not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError('Invalid GO enrichment p-values')
    # Independent hypergeometric-tail check of all displayed Fisher enrichment p-values.
    independently_computed = hypergeom.sf(d.n_hits - 1, d.n_background, d.n_annotated, d.n_selected)
    if not np.allclose(independently_computed, pvalues, rtol=1e-9, atol=1e-14):
        raise ValueError('GO enrichment p-values disagree with contingency counts')
    d['source_set'] = d.role.map({'activator_like': 'positive-corr', 'repressor_like': 'negative-corr'})
    d['minus_log10_p'] = -np.log10(d.p_value.clip(lower=1e-300))
    d = d.sort_values(keys)
    d.to_csv(dest / f'{stem}.source.csv', index=False)
    lookup = d.set_index(keys)

    # Different hues identify data; saturation/lightness identify prediction method.
    palettes = {'eqtl': .59, 'mpra': .075}
    colors = {(dataset, method): colorsys.hls_to_rgb(palettes[dataset], .43 if method == 'seq2kd' else .76,
                                                   .72 if method == 'seq2kd' else .38)
              for dataset in datasets for method in methods}
    combos = list(itertools.product(datasets, methods))
    labels = {('eqtl', 'seq2kd'): 'eQTL · Kd', ('eqtl', 'pwm'): 'eQTL · PWM motif score',
              ('mpra', 'seq2kd'): 'MPRA · Kd', ('mpra', 'pwm'): 'MPRA · PWM motif score'}
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 11,
                         'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1, .85, .85]})
        fig.subplots_adjust(left=.075, right=.94, bottom=.12, top=.90, hspace=.85)
        width = .175
        centers = np.array([i * 2.6 + j for i in range(len(percentages)) for j in range(2)])
        threshold = -np.log10(.05)
        for panel, ax in enumerate(axes):
            role = roles[panel % 2]
            is_count = panel >= 2
            title = ['Positive-corr (r > 0, regression P < 0.05)', 'Negative-corr (r < 0, regression P < 0.05)'][panel % 2]
            if is_count:
                title += ' · TFs matching target GO set'
            highest = 0.
            for index, combo in enumerate(combos):
                values = [float(lookup.loc[(*combo, p, role, term), 'n_hits' if is_count else 'minus_log10_p'])
                          for p in percentages for term in terms]
                highest = max(highest, max(values))
                bars = ax.bar(centers + (index - 1.5) * width, values, width=width * .92,
                       color=colors[combo], edgecolor='#30343b', linewidth=.45,
                       hatch='///' if combo[1] == 'pwm' else None, zorder=3)
                if is_count:
                    ax.bar_label(bars, labels=[str(int(v)) for v in values], padding=3, fontsize=8, rotation=90)
            if not is_count:
                ax.axhline(threshold, color='#9b4141', linewidth=1, linestyle=(0, (4, 3)), zorder=2)
                ax.text(1.006, threshold, 'P = 0.05', transform=ax.get_yaxis_transform(),
                    ha='left', va='bottom', color='#9b4141', fontsize=9)
            ax.set_ylim(0, max(highest * (1.28 if is_count else 1.16), 1 if is_count else threshold * 1.25))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=is_count))
            ax.grid(axis='y', color='#e5e7eb', linewidth=.65, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylabel('Number of matching TFs' if is_count else '−log₁₀(enrichment P-value)')
            ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
            ax.set_xticks(centers, ['Activator GO\nGO:0001228', 'Repressor GO\nGO:0001227'] * len(percentages))
            ax.tick_params(axis='x', length=0, pad=8, labelsize=10)
            for i, percent in enumerate(percentages):
                midpoint = i * 2.6 + .5
                ax.text(midpoint, -.255, f'p = {percent:g}%', transform=ax.get_xaxis_transform(),
                        ha='center', va='top', fontsize=12, fontweight='bold')
                if i:
                    ax.axvline(i * 2.6 - .8, color='#d1d5db', linewidth=.75, zorder=1)
            ax.set_xlim(centers.min() - .65, centers.max() + .65)
        fig.suptitle('GO enrichment across binding thresholds' + (' · Overlap excluded from GO sets' if exclusive else ''), fontsize=18, y=.98)
        handles = [Patch(facecolor=colors[c], edgecolor='#30343b', linewidth=.45,
                         hatch='///' if c[1] == 'pwm' else None, label=labels[c]) for c in combos]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .95),
                   ncol=4, frameon=False, fontsize=11, handlelength=2.3, columnspacing=2)
        fig.text(.075, .035,
                 'Aggregate effects · One-sided Fisher enrichment · Raw P-values · Panel y-scales differ\n'
                 'Counts = selected TFs intersect target GO set (n_hits); positions match the enrichment bars above.\n'
                 + ('Target GO sets: activator-only / repressor-only; overlapping TFs remain in the tested background.\n' if exclusive else 'Original GO sets: TFs annotated to both terms can contribute to both targets.\n')
                 + 'p%: binding-site selection threshold; hue: dataset; saturated/solid: Kd; pale/hatched: PWM motif score.',
                 fontsize=10, color='#4b5563', linespacing=1.6)
        fig.savefig(dest / f'{stem}.pdf', bbox_inches='tight')
        fig.savefig(dest / f'{stem}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    validation = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'displayed_rows': len(d), 'datasets': datasets, 'methods': methods,
                  'percentages': [float(x) for x in percentages], 'group': 'aggregate',
                  'p_value_column': 'p_value', 'reference_p': .05, 'zero_p_floor': 1e-300,
                  'independent_hypergeometric_check_passed': True,
                  'exclusive_go_sets': exclusive, 'count_column': 'n_hits',
                  'background_policy': 'unchanged tested GOALL-annotated TFs',
                  'max_abs_p_check_difference': float(np.max(np.abs(independently_computed - pvalues))),
                  'y_scale': 'independent per panel; both start at zero',
                  'colors_rgb': {f'{a}/{b}': list(c) for (a, b), c in colors.items()}}
    (dest / f'{stem}.validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Exported GO comparison:', len(d), 'verified rows;', dest / f'{stem}.pdf', flush=True)
    if not exclusive and include_exclusive:
        make_go_enrichment_figures(out, exclusive=True)
