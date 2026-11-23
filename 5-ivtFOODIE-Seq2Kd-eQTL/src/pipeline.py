#!/usr/bin/env python3
"""Source-backed, resumable Seq2Kd / PWM association pipeline. All writes stay here."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import linregress, fisher_exact


def output_path(p):
    p = Path(p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def config(args):
    c = json.loads(Path(args.config).read_text())
    for key in ['binding_rule', 'eqtl_beta_rule', 'seed', 'mpra_sample_size', 'batch_size']:
        value = getattr(args, key, None)
        if value is not None:
            c[key] = value
    if args.percentages:
        c['percentages'] = args.percentages
    if args.gpus is not None:
        c['gpus'] = args.gpus
    if not all(0 < p <= 100 for p in c['percentages']):
        raise ValueError('percentages must be in (0,100]')
    return c


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def variant_parts(ids, sep='_'):
    z = ids.str.split(sep, expand=True)
    return pd.DataFrame({'chrom': 'chr' + z[0].str.removeprefix('chr'),
                         'pos': z[1].astype(int), 'ref': z[2], 'alt': z[3]})


def is_snv(d):
    return d.ref.isin(list('ACGT')) & d.alt.isin(list('ACGT')) & (d.ref != d.alt)


def rebuild_eqtl(c, dest):
    """Exactly the notebook's variant-global PIP gate, before SNP-gene aggregation."""
    import pyarrow.parquet as pq
    base = Path(c.get('gtex_dir') or Path(c['source_root']) / 'important_data/promoterAI/eQTL')
    valid = set()
    fine_root = base / 'GTEx_v10_SuSiE_eQTL'
    fine_files = sorted(fine_root.rglob('*.parquet')) or sorted(fine_root.rglob('*.txt.gz'))
    if not fine_files:
        raise FileNotFoundError('No extracted GTEx v10 SuSiE files')
    for p in fine_files:
        columns = ['variant_id', 'pip']
        d = (pd.read_parquet(p, columns=columns) if p.suffix == '.parquet'
             else pd.read_csv(p, sep='\t', usecols=columns))
        valid.update(d.loc[d.pip > .9, 'variant_id'])
    frames = []
    pairs_root = base / 'GTEx_Analysis_v10_eQTL'
    pair_files = (sorted(pairs_root.rglob('*.signif_variant_gene_pairs.parquet'))
                  or sorted(pairs_root.rglob('*.signif_variant_gene_pairs.txt.gz'))
                  or sorted(pairs_root.rglob('*.parquet')))
    if not pair_files:
        raise FileNotFoundError('No extracted GTEx v10 significant variant-gene pair files')
    for p in pair_files:
        print('GTEx', p.name, flush=True)
        columns = ['variant_id', 'gene_id', 'slope', 'tss_distance']
        if p.suffix == '.parquet':
            batches = (b.to_pandas() for b in pq.ParquetFile(p).iter_batches(
                batch_size=250000, columns=columns))
        else:
            batches = pd.read_csv(p, sep='\t', usecols=columns, chunksize=250000)
        for d in batches:
            d = d[d.variant_id.isin(valid)].copy()
            d['tissue'] = p.name.split('.v10.')[0]
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(['variant_id', 'gene_id'], sort=True).agg(
        tissue=('tissue', 'nunique'), avg_slope=('slope', 'mean'),
        tss_distance=('tss_distance', 'mean')).reset_index()
    g = g[(g.tissue >= 2) & (g.tss_distance.abs() <= 500)]
    g.to_csv(dest / 'eqtl_fine_mapped_rebuilt.csv', index=False)
    save_json({'pip_rule': '>0.9', 'pip_gate': 'variant present in any SuSiE tissue/gene',
               'susie_files': len(fine_files), 'pair_files': len(pair_files),
               'high_pip_variants': len(valid), 'selected_snp_gene_rows_before_snv': len(g)},
              dest / 'eqtl_raw_audit.json')
    return g


def sequences(d, genome, audit):
    import pysam
    # Existing .fai required: never create or update an index beside the source.
    if not Path(str(genome) + '.fai').is_file():
        raise FileNotFoundError(f'Missing read-only FASTA index {genome}.fai')
    rows, excluded = [], []
    with pysam.FastaFile(str(genome)) as f:
        for r in d.itertuples(index=False):
            # Legacy SNP sequences contain 20 bases left + SNP + 19 bases right (40 nt).
            try:
                seq = f.fetch(r.chrom, r.pos - 21, r.pos + 19).upper()
                if len(seq) != 40 or any(b not in 'ACGT' for b in seq):
                    raise ValueError('boundary_or_non_ACGT')
                if seq[20] != r.ref:
                    raise ValueError(f'reference_mismatch:{seq[20]}!={r.ref}')
                rows.append((r.variant_id, r.chrom, r.pos, r.ref, r.alt, seq,
                             seq[:20] + r.alt + seq[21:]))
            except (ValueError, KeyError) as e:
                excluded.append((r.variant_id, str(e)))
    pd.DataFrame(excluded, columns=['variant_id', 'reason']).to_csv(audit, index=False)
    return pd.DataFrame(rows, columns=['variant_id', 'chrom', 'pos', 'ref', 'alt', 'ref_seq', 'alt_seq'])


def prepare(c, out, datasets=('eqtl',)):
    dest = output_path(out / 'prepared')
    audit = {}
    if 'eqtl' in datasets:
        g = rebuild_eqtl(c, dest)
        z = variant_parts(g.variant_id)
        g = pd.concat([g.reset_index(drop=True), z.reset_index(drop=True)], axis=1)
        before = len(g)
        g = g[is_snv(g)].copy()
        eqseq = sequences(g.drop_duplicates('variant_id'), Path(c['genome_dir']) / 'hg38.fa', dest / 'eqtl_excluded_sequences.csv')
        g = g[g.variant_id.isin(eqseq.variant_id)].copy()
        g['outcome'] = g.avg_slope
        g['group'] = 'aggregate'
        g['observation_id'] = g.variant_id + '|' + g.gene_id
        if c['eqtl_beta_rule'] == 'legacy_last_gene':
            # Explicit old dict(zip(...)) last-gene overwrite, while retaining original row multiplicity.
            last = g.drop_duplicates('variant_id', keep='last').set_index('variant_id').avg_slope
            g['outcome'] = g.variant_id.map(last)
        g.to_csv(dest / 'eqtl_observations.csv', index=False)
        eqseq.to_csv(dest / 'eqtl_variants.csv', index=False)

        audit['eqtl'] = {'input_rows': before, 'observations': len(g), 'variants': len(eqseq),
                         'beta_rule': c['eqtl_beta_rule']}
    if 'mpra' in datasets:
        # Collapse exact assay duplicates caused by repeated GWAS/cohort/trait annotation.
        cols = ['variant', 'cell_type', 'library', 'log2Skew']
        seen = []
        raw_rows, non_snv = 0, 0
        for d in pd.read_csv(c['mpra'], sep='\t', usecols=cols, chunksize=200000):
            raw_rows += len(d)
            z = variant_parts(d.variant, sep=':')
            valid = is_snv(z)
            non_snv += int((~valid).sum())
            d = d[valid & np.isfinite(pd.to_numeric(d.log2Skew, errors='coerce'))]
            seen.append(d.drop_duplicates(cols))
        assays = pd.concat(seen, ignore_index=True).drop_duplicates(cols)
        # Equal weight per library within cell, then equal weight per cell within SNP.
        assay = assays.groupby(['variant', 'cell_type', 'library'], sort=True).log2Skew.mean().reset_index()
        cells = assay.groupby(['variant', 'cell_type'], sort=True).log2Skew.mean().reset_index()
        agg = cells.groupby('variant', sort=True).log2Skew.mean()
        pool = np.array(sorted(agg.index))
        selected = np.random.RandomState(c['seed']).choice(pool, size=min(c['mpra_sample_size'], len(pool)), replace=False)
        selected = sorted(selected)
        vs = pd.DataFrame({'variant_id': selected})
        vs = pd.concat([vs, variant_parts(vs.variant_id, sep=':')], axis=1)
        mpseq = sequences(vs, Path(c['genome_dir']) / 'hg19.fa', dest / 'mpra_excluded_sequences.csv')
        # Sample before genome QC; excluded variants are reported, never silently resampled.
        cells = cells[cells.variant.isin(mpseq.variant_id)].rename(columns={'variant': 'variant_id', 'cell_type': 'group', 'log2Skew': 'outcome'})
        aggd = pd.DataFrame({'variant_id': agg.index, 'outcome': agg.values})
        aggd = aggd[aggd.variant_id.isin(mpseq.variant_id)].copy()
        aggd['group'] = 'aggregate'
        obs = pd.concat([aggd, cells], ignore_index=True)
        obs['observation_id'] = obs.variant_id + '|' + obs.group
        obs.to_csv(dest / 'mpra_observations.csv', index=False)
        mpseq.to_csv(dest / 'mpra_variants.csv', index=False)
        pd.DataFrame({'variant_id': selected}).to_csv(dest / 'mpra_sampled_ids.csv', index=False)

        audit['mpra'] = {'input_rows': raw_rows, 'non_snv_rows': non_snv,
                         'unique_assays': len(assay), 'eligible': len(pool),
                         'sampled': len(selected), 'variants': len(mpseq), 'seed': c['seed']}

    proteins = pd.read_csv(Path(c['model_dir']) / c['proteins'])
    proteins['tf'] = proteins.uniprot.str.split('_').str[-1]
    proteins['protein_id'] = proteins.uniprot
    if proteins.protein_id.duplicated().any():
        raise ValueError('Non-unique protein IDs')
    proteins.to_csv(dest / 'proteins.csv', index=False)
    audit['proteins'] = len(proteins)
    save_json(audit, dest / 'audit.json')


def pwm_scores(seqs, pwm):
    """Original HOCOMOCO log-odds matrix; max additive score over windows/both strands."""
    idx = np.array([[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[x] for x in s] for s in seqs])
    width = pwm.shape[0]
    if width > idx.shape[1]:
        raise ValueError('PWM longer than DNA')
    scores = np.full(len(idx), -np.inf)
    for start in range(idx.shape[1] - width + 1):
        windows = idx[:, start:start + width]
        for matrix in (pwm, pwm[::-1, ::-1]):
            scores = np.maximum(scores, matrix[np.arange(width)[None, :], windows].sum(axis=1))
    return scores


def score_pwm(c, out, datasets=('eqtl',)):
    dest = output_path(out / 'scores/pwm')
    go = pd.read_csv(Path(c['source_root']) / 'important_data/promoterAI/human_tf_add_GO_term.tsv', sep='\t').fillna('')
    # HOCOMOCO names use UniProt entry prefixes (ANDR, TFE2...), not always gene symbols.
    name_to_gene = dict(zip(go['Entry Name'].str.split('_').str[0], go['Gene Names'].str.split().str[0]))
    manifest = []
    inputs = {d: pd.read_csv(out / f'prepared/{d}_variants.csv') for d in datasets}
    for p in sorted(Path(c['pwm_dir']).glob('*.pwm')):
        if p.name.split('.')[2] != '0':
            continue
        motif = p.name.split('.')[0]
        tf = name_to_gene.get(motif, motif)
        if not isinstance(tf, str) or not tf:
            tf = motif
        pwm = np.loadtxt(p, skiprows=1, ndmin=2)
        if pwm.shape[1] != 4 or not np.isfinite(pwm).all():
            raise ValueError(f'Invalid PWM {p}')
        manifest.append({'motif_id': p.stem, 'tf': tf, 'motif_name': motif, 'path': str(p), 'sha256': sha256(p)})
        for dataset, variants in inputs.items():
            if pwm.shape[0] > variants.ref_seq.str.len().min():
                continue
            d = variants[['variant_id']].copy()
            d['ref_score'] = pwm_scores(variants.ref_seq, pwm)
            d['alt_score'] = pwm_scores(variants.alt_seq, pwm)
            d['tf'] = tf
            d['feature_id'] = p.stem
            d.to_parquet(dest / f'{dataset}.{p.stem}.parquet', index=False)
    pd.DataFrame(manifest).to_csv(out / 'prepared/pwm_manifest.csv', index=False)
    print('PWM scored', len(manifest), 'primary motifs', flush=True)


def import_eqtl(c, out):
    """Reuse the notebook-identified v20_class_1 scores; never run full eQTL inference."""
    dest = output_path(out / 'scores/seq2kd')
    source = Path(c['source_root']) / 'important_data/promoterAI/eQTL/sk_fine_mapped_v20_all_tf.add_position.drop_seq.csv'
    variants = pd.read_csv(out / 'prepared/eqtl_variants.csv')
    obs = pd.read_csv(out / 'prepared/eqtl_observations.csv')
    eligible = set(variants.variant_id)
    rows = {}
    count = 0
    for d in pd.read_csv(source, usecols=['uniprot', 'tf', 'wt_predicted_value', 'mut_predicted_value', 'chrom', 'pos', 'ref', 'alt'],
                         dtype={'chrom': str}, chunksize=200000):
        count += len(d)
        d['variant_id'] = 'chr' + d.chrom.str.removeprefix('chr') + '_' + d.pos.astype(str) + '_' + d.ref + '_' + d.alt + '_b38'
        d = d[d.variant_id.isin(eligible)].copy()
        for tf, sub in d.groupby('tf', sort=False):
            rows.setdefault(tf, []).append(sub)
    manifest, audit = [], []
    for tf, chunks in sorted(rows.items()):
        d = pd.concat(chunks, ignore_index=True)
        # Original table strips fragment suffixes; order within each TF preserves input
        # SNP-gene order. Pair repeated rows to SNP-gene IDs with explicit occurrence keys.
        expected_order = obs.variant_id.tolist()
        repetitions = len(d) // len(obs)
        original_count = len(d)
        source_ids = sorted(d.uniprot.unique())
        # Keep SNP-gene observations, averaging ref and alt separately across fragments.
        if repetitions >= 1 and len(d) == repetitions * len(obs) and d.variant_id.tolist() == expected_order * repetitions:
            d['observation_id'] = obs.observation_id.tolist() * repetitions
            score = d.groupby(['variant_id', 'observation_id'], sort=True).agg(
                ref_score=('wt_predicted_value', 'mean'), alt_score=('mut_predicted_value', 'mean')).reset_index()
        else:
            # A reordered cache must not silently map beta/gene by row position.
            raise ValueError(f'Historical score order/multiplicity cannot be matched to SNP-gene rows: {tf}')
        score['tf'], score['feature_id'] = tf, tf
        if set(score.variant_id) != eligible or set(score.observation_id) != set(obs.observation_id):
            raise ValueError(f'Incomplete historical eQTL scores: {tf}')
        score.to_parquet(dest / f'eqtl.{tf}.parquet', index=False)
        manifest.append({'tf': tf, 'feature_id': tf, 'n_fragments': repetitions})
        audit.append({'tf': tf, 'source_rows': original_count, 'snp_gene_rows': len(score),
                      'n_fragments': repetitions, 'source_uniprot': ';'.join(source_ids)})
    pd.DataFrame(manifest).to_csv(out / 'prepared/eqtl_score_manifest.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'prepared/eqtl_import_counts.csv', index=False)
    save_json({'source': str(source), 'source_rows': count, 'source_sha256': sha256(source),
               'checkpoint_identity_evidence': 'promoterAI.ipynb cell 218 identifies eQTL/tf2corr_pvalue_dict.pkl as v20_class_1',
               'tf_count': len(manifest), 'fragment_rule': 'per-observation ref/alt means'},
              out / 'prepared/eqtl_score_provenance.json')
    print('Imported historical v20_class_1 eQTL scores for', len(manifest), 'TFs', flush=True)


def bh(p):
    p = np.asarray(p, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return result


def regress(d, obs, percent, rule, minimum):
    # Threshold over unique variants, so gene/cell multiplicity cannot weight binding eligibility.
    threshold_data = d.groupby('variant_id', sort=True)[['ref_score', 'alt_score']].mean()
    cut = np.quantile(np.r_[threshold_data.ref_score, threshold_data.alt_score] if rule == 'combined' else threshold_data.ref_score, 1 - percent / 100)
    bound = d[(d.ref_score >= cut) | (d.alt_score >= cut)].copy()
    bound['delta_score'] = bound.alt_score - bound.ref_score
    keys = ['variant_id', 'observation_id'] if 'observation_id' in bound else ['variant_id']
    joined = obs.merge(bound[keys + ['delta_score']], on=keys, validate='one_to_one' if len(keys) == 2 else 'many_to_one')
    rows = []
    for group, o in obs.groupby('group', sort=True):
        z = joined[joined.group == group]
        z = z[np.isfinite(z.delta_score) & np.isfinite(z.outcome)]
        row = dict(group=group, percent=percent, threshold=float(cut), n=len(z),
                   n_snvs=z.variant_id.nunique(), n_bound_snvs=bound.variant_id.nunique(),
                   n_bound_fragment_snvs=len(bound), status='ok')
        if len(z) < minimum:
            row['status'] = 'too_few_observations'
        elif z.delta_score.nunique() < 2:
            row['status'] = 'constant_predictor'
        elif z.outcome.nunique() < 2:
            row['status'] = 'constant_outcome'
        else:
            r = linregress(z.delta_score, z.outcome)
            row.update(slope=r.slope, intercept=r.intercept, r_value=r.rvalue,
                       p_value=r.pvalue, std_err=r.stderr)
        rows.append(row)
    return rows


def analyze(c, out, datasets=('eqtl', 'mpra')):
    dest = output_path(out / 'analysis')
    all_rows = []
    for dataset, method in itertools.product(datasets, ['seq2kd', 'pwm']):
        obs = pd.read_csv(out / f'prepared/{dataset}_observations.csv')
        files = sorted((out / f'scores/{method}').glob(f'{dataset}.*.parquet'))
        if not files:
            raise FileNotFoundError(f'Missing {dataset}/{method} scores')
        historical = dataset == 'eqtl' and method == 'seq2kd' and c.get('historical_scores', False)
        manifest_path = ('eqtl_score_manifest.csv' if historical else
                         'proteins.csv' if method == 'seq2kd' else 'pwm_manifest.csv')
        manifest = pd.read_csv(out / 'prepared' / manifest_path)
        idcol = 'feature_id' if historical else ('protein_id' if method == 'seq2kd' else 'motif_id')
        actual = {str(f) for f in files}
        # Coverage is a completion gate; a smoke run cannot produce an apparently complete report.
        expected = {str(out / f'scores/{method}/{dataset}.{feature}.parquet') for feature in manifest[idcol]}
        if expected != actual:
            raise ValueError(f'{dataset}/{method} score coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}')
        for tf, features in manifest.groupby('tf', sort=True):
            d = pd.concat([pd.read_parquet(out / f'scores/{method}/{dataset}.{feature}.parquet') for feature in features[idcol]], ignore_index=True)
            score_keys = ['variant_id', 'feature_id'] + (['observation_id'] if 'observation_id' in d else [])
            if d.duplicated(score_keys).any():
                raise ValueError(f'Duplicate scored fragment-variants: {tf}')
            if not set(obs.variant_id).issubset(set(d.variant_id)):
                raise ValueError(f'Incomplete scores: {tf}')
            feature_ids = ';'.join(sorted(d.feature_id.unique()))
            n_features = d.feature_id.nunique()
            # User-confirmed: average ref/alt across protein fragments BEFORE binding/ranking.
            group_keys = ['variant_id'] + (['observation_id'] if 'observation_id' in d else [])
            d = d.groupby(group_keys, sort=True)[['ref_score', 'alt_score']].mean().reset_index()
            for percent in c['percentages']:
                for r in regress(d, obs, percent, c['binding_rule'], c['min_observations']):
                    r.update(dataset=dataset, method=method, tf=tf,
                             feature_id=feature_ids, n_features=n_features)
                    all_rows.append(r)
    result = pd.DataFrame(all_rows)
    for column in ['p_value', 'r_value']:
        if column not in result:
            result[column] = np.nan
    for _, ix in result.groupby(['dataset', 'method', 'group', 'percent']).groups.items():
        result.loc[ix, 'q_value'] = bh(result.loc[ix, 'p_value'])
    result['role'] = np.select([(result.p_value < .05) & (result.r_value > 0),
                                (result.p_value < .05) & (result.r_value < 0)],
                               ['activator_like', 'repressor_like'], default='not_significant')
    result.to_csv(dest / 'regressions.csv', index=False)
    # TF-level collections are unions of significant features; keep dual-role TFs explicit.
    memberships = result[result.role != 'not_significant'][['dataset', 'method', 'group', 'percent', 'tf', 'role']].drop_duplicates()
    memberships.to_csv(dest / 'tf_memberships.csv', index=False)
    rows = []
    for key, sub in result.groupby(['dataset', 'method', 'group']):
        for role in ['activator_like', 'repressor_like']:
            sets = {p: set(sub.loc[(sub.percent == p) & (sub.role == role), 'tf']) for p in c['percentages']}
            for p, q in itertools.combinations(c['percentages'], 2):
                a, b = sets[p], sets[q]
                rows.append(dict(zip(['dataset', 'method', 'group'], key), role=role, p1=p, p2=q,
                    n1=len(a), n2=len(b), intersection=len(a & b), union=len(a | b),
                    jaccard=len(a & b) / len(a | b) if a | b else np.nan,
                    shared=';'.join(sorted(a & b)), only_p1=';'.join(sorted(a - b)), only_p2=';'.join(sorted(b - a))))
    if rows:
        pd.DataFrame(rows).to_csv(dest / 'percent_set_comparisons.csv', index=False)
    compare = []
    for key, sub in result.groupby(['dataset', 'group', 'percent']):
        kd, pwm = sub[sub.method == 'seq2kd'], sub[sub.method == 'pwm']
        common = set(kd.tf) & set(pwm.tf)
        for role in ['activator_like', 'repressor_like']:
            a = set(kd.loc[kd.role == role, 'tf']) & common
            b = set(pwm.loc[pwm.role == role, 'tf']) & common
            compare.append(dict(zip(['dataset', 'group', 'percent'], key), role=role,
                shared_tf_universe=len(common), kd=len(a), pwm=len(b), intersection=len(a & b),
                jaccard=len(a & b) / len(a | b) if a | b else np.nan,
                shared=';'.join(sorted(a & b)), kd_only=';'.join(sorted(a - b)), pwm_only=';'.join(sorted(b - a))))
    pd.DataFrame(compare).to_csv(dest / 'seq2kd_pwm_comparisons.csv', index=False)
    save_json(c, dest / 'analysis_config.json')
    print('Analyzed', len(result), 'regressions', flush=True)


def go_analysis(c, out):
    dest = output_path(out / 'analysis')
    go = pd.read_csv(Path(c['source_root']) / 'important_data/promoterAI/human_tf_add_GO_term.tsv', sep='\t').fillna('')
    go['tf'] = go['Gene Names'].str.split().str[0]
    go['activate_count'] = go['Gene Ontology (GO)'].str.count('activator|activation')
    go['repress_count'] = go['Gene Ontology (GO)'].str.count('repressor|repression')
    go['keyword_role'] = np.select([(go.activate_count > 0) & (go.repress_count == 0),
                                     (go.repress_count > 0) & (go.activate_count == 0)],
                                    ['activator', 'repressor'], default='mixed_or_unannotated')
    result = pd.read_csv(dest / 'regressions.csv')
    result.merge(go[['tf', 'activate_count', 'repress_count', 'keyword_role', 'Gene Ontology IDs']].drop_duplicates('tf'),
                 on='tf', how='left', validate='many_to_one').to_csv(dest / 'go_keyword_comparison.csv', index=False)
    # Export org.Hs.eg.db GOALL once; includes ancestor annotations, matching enrichGO semantics.
    goall_path = out / 'prepared/human_go_all.csv'
    if not goall_path.exists():
        env = os.environ.copy()
        env['R_USER'] = str(ROOT / '.cache/R')
        subprocess.run([c['rscript'], '--vanilla', str(ROOT / 'export_go.R'), str(goall_path)], check=True, env=env)
    annotations = pd.read_csv(goall_path).dropna().drop_duplicates()
    rows = []
    targets = ['GO:0001228', 'GO:0001227']
    for key, sub in result.groupby(['dataset', 'method', 'group', 'percent']):
        # Tested TFs with an org.Hs.eg.db GO annotation: explicit assay-matched background.
        universe = set(sub.loc[sub.status == 'ok', 'tf']) & set(annotations.SYMBOL)
        for role in ['activator_like', 'repressor_like']:
            selected = set(sub.loc[sub.role == role, 'tf']) & universe
            for term in targets:
                annotated = set(annotations.loc[annotations.GOALL == term, 'SYMBOL']) & universe
                hits = selected & annotated
                table = [[len(hits), len(selected - annotated)],
                         [len(annotated - selected), len(universe - (selected | annotated))]]
                odds, p = fisher_exact(table, alternative='greater') if universe and selected else (np.nan, np.nan)
                rows.append(dict(zip(['dataset', 'method', 'group', 'percent'], key), role=role,
                    term=term, n_background=len(universe), n_selected=len(selected),
                    n_annotated=len(annotated), n_hits=len(hits), odds_ratio=odds,
                    p_value=p, hits=';'.join(sorted(hits))))
    enrich = pd.DataFrame(rows)
    for _, ix in enrich.groupby(['dataset', 'method', 'group', 'percent']).groups.items():
        enrich.loc[ix, 'q_value_four_tests'] = bh(enrich.loc[ix, 'p_value'])
    enrich.to_csv(dest / 'go_target_enrichment.csv', index=False)


def infer(c, out, limit=None, datasets=('eqtl',)):
    logdir = output_path(out / 'logs')
    processes = []
    workers = len(c['gpus'])
    if workers == 0:
        raise ValueError('At least one GPU required')
    runtime = out / 'runtime_config.json'
    save_json(c, runtime)
    try:
        for shard, gpu in enumerate(c['gpus']):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['OMP_NUM_THREADS'] = '2'
            env['MKL_NUM_THREADS'] = '2'
            log = open(logdir / f'inference_gpu{gpu}.log', 'w')
            command = [c['python'], '-u', str(ROOT / 'infer_seq2kd.py'), '--config', str(runtime),
                       '--output', str(out), '--shard', str(shard), '--shards', str(workers),
                       '--datasets', *datasets]
            if limit is not None:
                command += ['--limit-proteins', str(limit)]
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            failed = [p.returncode for p, _ in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Inference worker failed: {failed}; inspect {logdir}')
            time.sleep(5)
        if any(p.returncode for p, _ in processes):
            raise RuntimeError(f'Inference failed; inspect {logdir}')
    finally:
        for p, log in processes:
            if p.poll() is None:
                p.terminate()
            log.close()


def summarize(c, out):
    from plots import make_go_enrichment_figures
    make_go_enrichment_figures(out, include_exclusive=c.get('threshold_sensitivity', False))
    d = pd.read_csv(out / 'analysis/regressions.csv')
    rows = d[d.group == 'aggregate'].groupby(
        ['dataset', 'method', 'percent', 'role']).size().rename('n_tf').reset_index()
    rows.to_csv(out / 'analysis/tf_counts.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'import-eqtl', 'pwm', 'infer', 'analyze', 'go', 'summarize', 'run'])
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--output', default=str(ROOT / 'results'))
    parser.add_argument('--percentages', '--p', type=float, nargs='+')
    parser.add_argument('--binding-rule', choices=['combined', 'legacy_ref'])
    parser.add_argument('--eqtl-beta-rule', choices=['snp_gene', 'legacy_last_gene'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--mpra-sample-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--historical-scores', action='store_true',
                        help='Use published historical GTEx scores for audit in a separate output directory')
    parser.add_argument('--threshold-sensitivity', action='store_true')
    parser.add_argument('--limit-proteins', type=int, help='Smoke test only, restrict protein count')
    parser.add_argument('--datasets', nargs='+', choices=['eqtl', 'mpra'], default=['eqtl'], help='Datasets to prepare, score and analyze')
    args = parser.parse_args()
    c, out = config(args), output_path(args.output)
    c['historical_scores'] = args.historical_scores
    c['threshold_sensitivity'] = args.threshold_sensitivity
    if args.threshold_sensitivity and args.percentages is None:
        c['percentages'] = [1, 5, 10, 30]
    if args.stage == 'import-eqtl' and not args.historical_scores:
        parser.error('import-eqtl requires --historical-scores and a separate audit output')
    save_json(c, out / 'config.json')
    stages = ['prepare', 'pwm', 'infer', 'analyze', 'go', 'summarize'] if args.stage == 'run' else [args.stage]
    for stage in stages:
        if stage == 'prepare':
            prepare(c, out, args.datasets)
        elif stage == 'pwm':
            score_pwm(c, out, args.datasets)
        elif stage == 'import-eqtl':
            import_eqtl(c, out)
        elif stage == 'infer':
            datasets = args.datasets
            if args.historical_scores and 'eqtl' in datasets:
                import_eqtl(c, out)
                datasets = [d for d in datasets if d != 'eqtl']
            if datasets:
                infer(c, out, args.limit_proteins, datasets)
        elif stage == 'analyze':
            analyze(c, out, args.datasets)
        elif stage == 'go':
            go_analysis(c, out)
        else:
            summarize(c, out)


if __name__ == '__main__':
    main()
