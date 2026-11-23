#!/usr/bin/env python3
"""Add top50/top100 to the requested mean AUC plots in an isolated branch.

Percentile cutoffs use ALL 50,823 promoter variants, as before. Only the two
clinical classes plotted in the requested figures require additional inference.
Old top1/5/10/30 results are retained and independently checked where recomputed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, digest, filehash, parquet, save_json, PROBOUND
from candidates import score_matrix
from motifs import encode

BASE = ROOT/'results'
RUN = BASE/'promoter_mean_top100'
OUT = BASE/'analysis/requested_median_pvalue_figures'
PCTS = [1, 5, 10, 30, 50, 100]
NAMES = ['02_promoter_streme_jaspar_top100_auc_mean.pdf',
         '03_promoter_methods_top100_auc_mean.pdf']
V = REF = ALT = BP = None


def scan(job):
    method, feature, tf, pid, mat, logsum = job
    path = RUN/'candidates'/method/(feature+'.parquet')
    if path.exists():
        return str(path)
    r = score_matrix(REF, mat, logsum)
    a = score_matrix(ALT, mat, logsum)
    binding = np.maximum(r, a)
    cutoffs = {pct: float(np.percentile(binding, 100-pct)) for pct in PCTS}
    d = V.loc[BP, ['variant_id']].copy()
    d['tf'] = tf
    d['protein_id'] = pid
    d['abs_delta'] = np.abs(a[BP]-r[BP])
    for pct in PCTS:
        d[f'top{pct}'] = binding[BP] >= cutoffs[pct]
    parquet(d, path)
    save_json(dict(cutoffs=cutoffs, universe=len(V), plotted_cohort=len(d),
                   feature=feature, tf=tf, protein_id=pid), path.with_suffix('.json'))
    return str(path)


def init_scan():
    global V, REF, ALT, BP
    V = pd.read_parquet(BASE/'prepared/promoter_variants.parquet')
    assert not V.variant_id.duplicated().any()
    REF, ALT = encode(V.ref_seq), encode(V.alt_seq)
    BP = V.label.isin(['Benign', 'Pathogenic']).to_numpy()
    return V


def jobs():
    result, neural = [], []
    for method, path, source in [('seq2kd_streme', BASE/'prepared/motifs.json', 'streme'),
                                  ('seq2kd_jaspar', BASE/'jaspar_coding/prepared/motifs.json', None)]:
        motifs = json.loads(path.read_text())
        if source:
            motifs = [m for m in motifs if m['source'] == source]
        for m in motifs:
            matrix = np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
            result.append((method, m['protein_id'], m['tf'], m['protein_id'], matrix, False))
            neural.append(dict(method=method, protein_id=m['protein_id'], tf=m['tf']))
        if method == 'seq2kd_jaspar':
            unique = {(m['tf'], m['motif_id']):m for m in motifs}
            for m in unique.values():
                matrix = np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
                result.append(('pwm_jaspar', m['tf']+'.'+m['motif_id'], m['tf'], '', matrix, False))
    # Same exact fit-selection rules as the original candidates.probound_jobs.
    proteins = pd.read_parquet(BASE/'prepared/proteins.parquet')
    allowed = set(proteins.tf.str.upper())
    for model in json.loads((PROBOUND/'MotifCentral.v1.0.0.json').read_text()):
        factors = sorted(set(f['gene_symbol'].upper() for f in model['metadata']['factors']
                             if str(f.get('tax_id')) == '9606'))
        if len(factors) != 1 or not (set(factors) & allowed):
            continue
        modes = model['modelSettings']['bindingModes']
        assert len(modes) == 1 and modes[0]['dinucleotideDistance'] == 0
        assert modes[0]['flankLength'] == 0 and not modes[0].get('singleStrand', False)
        matrix = np.array(model['coefficients']['bindingModes'][0]['mononucleotide']).reshape(-1,4)
        if len(matrix) > 40:
            continue
        tf = factors[0]
        feature = tf+'.'+str(model['metadata']['fit_id'])
        result.append(('probound', feature, tf, '', matrix, True))
    assert len({(j[0], j[1]) for j in result}) == len(result)
    return result, neural


def prepare(workers):
    for sub in ['prepared', 'qc', 'cache', 'scores', 'tasks', 'locks', 'logs', 'variant_summaries']:
        (RUN/sub).mkdir(parents=True, exist_ok=True)
    for method in ['seq2kd_streme', 'seq2kd_jaspar', 'pwm_jaspar', 'probound']:
        (RUN/'candidates'/method).mkdir(parents=True, exist_ok=True)
    if not (RUN/'qc/original_pdf_hashes.json').exists():
        save_json({str(p):filehash(p) for p in OUT.glob('*.pdf')}, RUN/'qc/original_pdf_hashes.json')
    for dest, src in [('prepared/promoter_variants.parquet', BASE/'prepared/promoter_variants.parquet'),
                       ('qc/inference_signature.json', BASE/'qc/inference_signature.json'),
                       ('embeddings', BASE/'embeddings')]:
        if not (RUN/dest).exists():
            (RUN/dest).symlink_to(src)
    init_scan()
    alljobs, neural = jobs()
    save_json(neural, RUN/'qc/neural_assignments.json')
    inputs = [BASE/'prepared/promoter_variants.parquet', BASE/'prepared/proteins.parquet',
              BASE/'prepared/motifs.json', BASE/'jaspar_coding/prepared/motifs.json',
              PROBOUND/'MotifCentral.v1.0.0.json']
    save_json(dict(input_sha256={str(p):filehash(p) for p in inputs}, pcts=PCTS,
                   universe=len(V), inferred_cohort=int(BP.sum()),
                   threshold_population='All promoter variants, independent of clinical labels',
                   neural_population='Only Benign and Pathogenic, exactly the plotted clinical classes',
                   score='mean across TFs; max across neural fragments/PWM motifs within TF; median across ProBound fits'),
              RUN/'qc/inputs.json')
    score_matrix(REF[:1], alljobs[0][4], False)
    score_matrix(REF[:1], alljobs[0][4], True)
    with mp.get_context('fork').Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(scan, alljobs)):
            if (i+1) % 100 == 0:
                print(f'Scanned {i+1}/{len(alljobs)} features', flush=True)
    annhash = filehash(BASE/'prepared/promoter_variants.parquet')
    signature = json.loads((BASE/'qc/inference_signature.json').read_text())
    proteins = pd.read_parquet(BASE/'prepared/proteins.parquet').set_index('protein_id')
    basefiles = {}
    for p in (BASE/'scores').glob('promoter.*.parquet'):
        pid = p.stem[len('promoter.'):].rsplit('.',1)[0]
        basefiles.setdefault(pid, []).append(p)
    assignment = {a['protein_id']:a for a in neural}
    target_ids = set(V.loc[BP, 'variant_id'])
    assert json.loads((BASE/'jaspar_seq2kd_promoter_all_thresholds/COMPLETE.json').read_text())['complete']

    def cache_one(pid):
        marker = RUN/'qc'/f'cache.{pid}.json'
        if marker.exists():
            return json.loads(marker.read_text())
        remaining = set(target_ids)
        pieces, lineage = [], []
        extended = BASE/'jaspar_seq2kd_promoter_all_thresholds/analysis/pair_scores'/f'{pid}.parquet'
        candidates = ([extended] if extended.exists() else []) + sorted(basefiles.get(pid, []))
        for path in candidates:
            if path.parent == BASE/'scores':
                meta = json.loads(path.with_suffix('.json').read_text())
                task = json.loads((BASE/'tasks'/(path.stem+'.json')).read_text())
                assert meta['task_signature'] == task['signature']
                assert all(meta['inference'][k] == value for k, value in signature.items())
                assert task['sequence'] == proteins.loc[pid, 'promoter_sequence']
                assert task['prepared_inputs']['promoter_variants.parquet'] == annhash
            d = pd.read_parquet(path, columns=['variant_id', 'protein_id', 'tf', 'ref_pkd', 'alt_pkd'])
            assert d.protein_id.eq(pid).all()
            d = d.loc[d.variant_id.isin(remaining)]
            if len(d):
                pieces.append(d)
                remaining -= set(d.variant_id)
                lineage.append(dict(path=str(path), sha256=filehash(path)))
        if pieces:
            d = pd.concat(pieces, ignore_index=True)
            assert not d.variant_id.duplicated().any()
            parquet(d, RUN/'cache'/f'{pid}.parquet')
        remaining = sorted(remaining)
        a = assignment[pid]
        for start in range(0, len(remaining), 2048):
            task = dict(task_id=f'promoter.{pid}.{start:05d}', stage='promoter', protein_id=pid,
                        tf=a['tf'], sequence=proteins.loc[pid, 'promoter_sequence'],
                        variant_ids=remaining[start:start+2048],
                        prepared_inputs={'promoter_variants.parquet':annhash},
                        candidate_files={})
            task['signature'] = digest(json.dumps(task, sort_keys=True))
            save_json(task, RUN/'tasks'/(task['task_id']+'.json'))
        result = dict(protein_id=pid, cached=len(target_ids)-len(remaining), new=len(remaining), lineage=lineage)
        save_json(result, marker)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(cache_one, sorted(assignment)):
            results.append(r)
            if len(results) % 100 == 0:
                print(f'Prepared {len(results)}/{len(assignment)} fragments', flush=True)
    summary = {k:sum(r[k] for r in results) for k in ['cached', 'new']}
    summary.update(proteins=len(assignment), tasks=len(list((RUN/'tasks').glob('*.json'))))
    save_json(summary, RUN/'qc/preparation.json')
    save_json(dict(queue_closed=True), RUN/'tasks/CLOSED')
    print(json.dumps(summary), flush=True)


def summarize(method):
    by_tf = {}
    taskmap = {}
    if method.startswith('seq2kd'):
        for tp in (RUN/'tasks').glob('*.json'):
            t = json.loads(tp.read_text())
            assert (RUN/'scores'/tp.name).exists(), f'Unfinished inference: {tp.name}'
            taskmap.setdefault(t['protein_id'], []).append(t)
    for path in sorted((RUN/'candidates'/method).glob('*.parquet')):
        d = pd.read_parquet(path)
        if method.startswith('seq2kd'):
            pid = path.stem
            scores = []
            if (RUN/'cache'/f'{pid}.parquet').exists():
                scores.append(pd.read_parquet(RUN/'cache'/f'{pid}.parquet'))
            for t in taskmap.get(pid, []):
                sp = RUN/'scores'/(t['task_id']+'.parquet')
                meta = json.loads(sp.with_suffix('.json').read_text())
                assert meta['task_signature'] == t['signature']
                signature = json.loads((RUN/'qc/inference_signature.json').read_text())
                assert all(meta['inference'][k] == value for k, value in signature.items())
                s = pd.read_parquet(sp)
                assert set(s.variant_id) == set(t['variant_ids'])
                scores.append(s)
            s = pd.concat(scores, ignore_index=True)
            assert not s.variant_id.duplicated().any()
            assert set(s.variant_id) == set(d.variant_id)
            d = d.drop(columns='abs_delta').merge(s[['variant_id', 'ref_pkd', 'alt_pkd']], on='variant_id', validate='one_to_one')
            d['abs_delta'] = (d.alt_pkd-d.ref_pkd).abs()
        tf = d.tf.iloc[0]
        by_tf.setdefault(tf, []).append(d[['variant_id', 'abs_delta']+[f'top{p}' for p in PCTS]])
    per_pct = {p:[] for p in PCTS}
    for tf, parts in by_tf.items():
        d = pd.concat(parts, ignore_index=True)
        for pct in PCTS:
            group = d.loc[d[f'top{pct}']].groupby('variant_id').abs_delta
            s = group.median() if method == 'probound' else group.max()
            per_pct[pct].append(s.rename('value').reset_index().assign(tf=tf))
    ann = pd.read_parquet(BASE/'prepared/promoter_variants.parquet')[['variant_id', 'label']]
    for pct in PCTS:
        tf = pd.concat(per_pct[pct], ignore_index=True)
        d = tf.groupby('variant_id', as_index=False).agg(mean=('value', 'mean'), n_tf=('tf', 'size'))
        d = d.merge(ann, on='variant_id', validate='one_to_one')
        assert np.isfinite(d['mean']).all()
        assert not d.variant_id.duplicated().any()
        if pct <= 30:
            if method in ['seq2kd_streme', 'probound']:
                oldpath = BASE/'analysis/variant_summaries'/f'promoter.{method}.top{pct}.mean.parquet'
            elif method == 'seq2kd_jaspar':
                oldpath = BASE/'jaspar_seq2kd_promoter_all_thresholds/analysis/variant_summaries'/f'promoter.{method}.top{pct}.mean.parquet'
            else:
                oldpath = BASE/'promoter_methods_all_thresholds/variant_summaries'/f'promoter.{method}.top{pct}.mean.parquet'
            old = pd.read_parquet(oldpath)
            old = old.loc[old.label.isin(['Benign', 'Pathogenic'])].set_index('variant_id').sort_index()
            check = d.set_index('variant_id').sort_index()
            assert old.index.equals(check.index), (method, pct, 'cohort mismatch')
            np.testing.assert_allclose(check['mean'], old['mean'], rtol=2e-5, atol=2e-6)
            assert (check.n_tf == old.n_tf).all()
            # Keep the exact published values for the four existing thresholds.
            d = old.reset_index()[['variant_id', 'mean', 'n_tf', 'label']]
        if pct == 100:
            assert len(d) == ann.label.isin(['Benign', 'Pathogenic']).sum()
        parquet(d, RUN/'variant_summaries'/f'promoter.{method}.top{pct}.mean.parquet')
    print(f'{method}: all six mean summaries validated', flush=True)


def render():
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score, average_precision_score
    import plot_requested_median_pvalues as plot
    assert not any((OUT/name).exists() for name in NAMES), 'Preserve existing PDFs'
    frozen = {str(p):filehash(p) for p in OUT.glob('*.pdf')}
    rows = []
    for method in ['seq2kd_streme', 'seq2kd_jaspar', 'probound', 'pwm_jaspar']:
        summarize(method)
        for pct in PCTS:
            endpoint = f'promoter.{method}.top{pct}.mean'
            d = pd.read_parquet(RUN/'variant_summaries'/(endpoint+'.parquet'))
            b, p = d.loc[d.label.eq('Benign'), 'mean'], d.loc[d.label.eq('Pathogenic'), 'mean']
            y = d.label.eq('Pathogenic')
            auc = float(roc_auc_score(y, d['mean']))
            test = mannwhitneyu(p, b, alternative='two-sided', method='auto')
            assert np.isclose(auc, test.statistic/(len(b)*len(p)), rtol=0, atol=1e-12)
            rows.append(dict(endpoint=endpoint, method=method, threshold_pct=pct,
                             benign_n=len(b), pathogenic_n=len(p), benign_median=float(b.median()),
                             pathogenic_median=float(p.median()), median_difference=float(p.median()-b.median()),
                             auc_pathogenic_higher=auc, raw_p=float(test.pvalue),
                             auprc_average_precision=float(average_precision_score(y, d['mean']))))
    stats = pd.DataFrame(rows)
    stats.to_csv(RUN/'mean_threshold_statistics.csv', index=False)
    motif = stats.loc[stats.method.isin(['seq2kd_streme', 'seq2kd_jaspar'])].copy()
    motif['label'] = [f'{r.method.split("_")[-1].upper()}\ntop {r.threshold_pct}%' for r in motif.itertuples()]
    # Preserve the score direction even if a new threshold has AUC below 0.5.
    if motif.auc_pathogenic_higher.min() >= .5:
        plot.paired_bars(motif, 'Promoter variants: STREME and JASPAR binding thresholds (mean across TFs)', NAMES[0], 23, use_auc=True)
    else:
        fig, ax = plot.plt.subplots(figsize=(23, 5.5))
        right = ax.twinx()
        x = np.arange(len(motif))
        a = ax.bar(x-.18, motif.median_difference, width=.30, color=plot.BLUE)
        b = right.bar(x+.18, motif.auc_pathogenic_higher, width=.30, color='#F7ECD4', edgecolor=plot.GOLD, hatch='///')
        ax.set_ylabel(r'Median $|\Delta pK_d|$: Pathogenic $-$ Benign', color=plot.BLUE)
        right.set_ylabel('AUC', color=plot.GOLD)
        right.set_ylim(0, 1)
        right.axhline(.5, color='#666666', linewidth=.8, linestyle='--')
        ax.set_xticks(x, plot.labels_with_n(motif))
        ax.bar_label(a, labels=[f'{v:.4f}' for v in motif.median_difference], padding=5, color=plot.BLUE, fontsize=10)
        right.bar_label(b, labels=[f'{v:.4f}' for v in motif.auc_pathogenic_higher], padding=5, color=plot.GOLD, fontsize=10)
        plot.decorate(ax)
        fig.suptitle('Promoter variants: STREME and JASPAR binding thresholds (mean across TFs)', y=.97, fontsize=14)
        fig.text(.5,.025,'P: Pathogenic; B: Benign. AUC positive class: Pathogenic; higher absolute effects predict Pathogenic.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.09,right=.92,bottom=.27,top=.79)
        fig.savefig(OUT/NAMES[0])
        plot.plt.close(fig)
    fig, axes = plot.plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)
    for ax, method, name, quantity, color in zip(axes,
            ['seq2kd_streme', 'probound', 'pwm_jaspar'], ['Seq2Kd–STREME', 'ProBound', 'PWM–JASPAR'],
            [r'Predicted binding affinity: $|\Delta pK_d|$',
             r'Relative binding affinity: $|\log_{10}(K_{a,\mathrm{ALT}}/K_{a,\mathrm{REF}})|$',
             r'PWM log2-odds: $|\Delta\mathrm{motif\ score}|$'], ['#376895', '#BB8426', '#63856B']):
        view = stats.loc[stats.method.eq(method)]
        x = np.arange(len(PCTS))
        method_aucs = stats.loc[~stats.method.eq('seq2kd_jaspar'), 'auc_pathogenic_higher']
        baseline = .5 if method_aucs.min() >= .5 else 0.
        bars = ax.bar(x, view.auc_pathogenic_higher-baseline, bottom=baseline, width=.58, color=color, zorder=3)
        ax.set_title(name+'\n'+quantity, fontsize=12, pad=16)
        ax.set_xticks(x, [f'top {r.threshold_pct}%\nP n={r.pathogenic_n:,}\nB n={r.benign_n:,}\n\n{r.median_difference:+.5f}' for r in view.itertuples()], fontsize=10)
        ax.set_ylim((.5, plot.auc_upper(method_aucs)) if baseline else (0, 1))
        if not baseline:
            ax.axhline(.5,color='#666666',linewidth=.8,linestyle='--')
        ax.bar_label(bars, labels=[f'{v:.4f}' for v in view.auc_pathogenic_higher], padding=5, fontsize=10)
        ax.spines['right'].set_visible(False)
        plot.decorate(ax)
    axes[0].set_ylabel('AUC')
    fig.suptitle('Promoter variants: methods and binding thresholds (mean across TFs)', y=.97, fontsize=15)
    fig.text(.5,.045,'P: Pathogenic; B: Benign. Values below sample counts: Median(P) − Median(B) of SNP mean scores.\n'
             'AUC positive class: Pathogenic; higher absolute effects predict Pathogenic; '
             +('AUC axis starts at 0.5.' if baseline else 'dashed line: AUC = 0.5.'), ha='center',fontsize=10)
    fig.subplots_adjust(left=.055,right=.985,bottom=.29,top=.77,wspace=.10)
    fig.savefig(OUT/NAMES[1])
    plot.plt.close(fig)
    assert frozen == {p:filehash(p) for p in frozen}
    original = json.loads((RUN/'qc/original_pdf_hashes.json').read_text())
    assert original == {p:filehash(p) for p in original}
    save_json(dict(complete=True, original_pdfs_unchanged=True, old_thresholds_reproduced=True,
                   pcts=PCTS, aggregation='mean across TFs', auc_crosscheck='Mann-Whitney U/(nP*nB)',
                   script_sha256=filehash(__file__), output_sha256={str(OUT/name):filehash(OUT/name) for name in NAMES}), RUN/'COMPLETE.json')
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'render'])
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    prepare(args.workers) if args.phase == 'prepare' else render()
