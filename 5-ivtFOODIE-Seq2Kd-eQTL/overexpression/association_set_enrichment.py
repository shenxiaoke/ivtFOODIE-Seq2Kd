#!/usr/bin/env python3
"""Extend frozen direction-set enrichment to PWM; preserve the 20261023 delivery."""
from pathlib import Path
import argparse
import collections
import hashlib
import itertools
import json
import time
import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ACT, REP = 'activator-like', 'repressor-like'
METHODS = ['seq2kd', 'pwm']
PCTS = [1, 5, 10, 30]
CFG = {'primary_suites': ['A', 'B-integrated'],
       'query_role_mapping': {'activator_like': ACT, 'repressor_like': REP,
                              'not_significant': 'not_significant'}}
INPUTS = []

def read(path, **kwargs):
    INPUTS.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.read_csv(path, **kwargs)

def save(frame, path):
    pd.DataFrame(frame).to_csv(path, sep='\t', index=False, na_rep='NA')

def bh(values):
    values = np.asarray(values, float)
    x = np.nan_to_num(values, nan=1.)
    order = np.argsort(x)
    q = np.empty(len(x))
    q[order] = np.minimum(1, np.minimum.accumulate((x[order]*len(x)/np.arange(1, len(x)+1))[::-1])[::-1])
    q[~np.isfinite(values)] = np.nan
    return q

def analyze(args):
    target = read(args.targets, sep='\t')
    targets = set(target.TF)
    hg = read(args.hgnc, sep='\t', keep_default_na=False)
    official = set(hg.symbol)
    lookup = {f: collections.defaultdict(set) for f in ['prev_symbol', 'alias_symbol', 'ensembl_gene_id', 'uniprot_ids']}
    for row in hg.to_dict('records'):
        for field, db in lookup.items():
            for value in str(row.get(field, '')).split('|'):
                if value:
                    db[value].add(row['symbol'])
    def canonical(s):
        s = str(s).strip()
        if s in official:
            return s, 'HGNC_current_symbol'
        for field, db in lookup.items():
            match = db.get(s, set())
            if len(match) == 1:
                return next(iter(match)), field+'_unique'
            if len(match) > 1:
                return None, field+'_ambiguous'
        return (s, 'historical_target_exact_no_current_HGNC') if s in targets else (None, 'unresolved')
    proteins = read(args.proteins, usecols=['tf','uniprot','protein_id'])
    accessions = collections.defaultdict(set)
    for row in proteins.itertuples():
        for value in [row.uniprot, row.protein_id]:
            accessions[row.tf].add(str(value).split('_')[0])
    mapping = {}
    def map_tf(s):
        if s not in mapping:
            tf, status = canonical(s)
            if tf is None and status == 'unresolved':
                possible = {canonical(a)[0] for a in accessions.get(s, set())}-{None}
                if len(possible) == 1:
                    tf, status = next(iter(possible)), 'model_accession_HGNC_unique'
                elif len(possible) > 1:
                    status = 'model_accession_ambiguous'
            mapping[s] = dict(raw_tf=s, TF=tf, mapping_status=status, in_target_1639=tf in targets)
        return mapping[s]['TF']

    audit = read(args.reference, sep='\t')
    refs = []
    metadata = ['reference_id','reference_batch','suite','version','analysis_tier','S_threshold']
    for _, sub in audit.groupby('reference_id', sort=True):
        assert sub.TF.nunique() == len(sub) == 1639
        eligible = set(sub.loc[sub.reference_eligible.eq(True), 'TF'])
        sets = {role: set(sub.loc[sub.reference_label.eq(role),'TF']) for role in [ACT, REP]}
        assert (sets[ACT] | sets[REP]) <= eligible and sets[ACT].isdisjoint(sets[REP])
        refs.append((sub.iloc[0][metadata].to_dict(), eligible, sets))

    cols1 = ['dataset','method','group','percent','tf','status','role','r_value','p_value']
    data = read(args.regressions, usecols=cols1)
    data = data[data.dataset.isin(['eqtl', 'mpra'])].copy()
    data['source'], data['assay'], data['stratum'] = 'legacy', data.dataset, data.group
    data = data[data.method.isin(METHODS)].copy()
    data['query_label'] = data.role.map(CFG['query_role_mapping'])
    expected = np.where(data.p_value.lt(.05)&data.r_value.gt(0),ACT,np.where(data.p_value.lt(.05)&data.r_value.lt(0),REP,'not_significant'))
    assert (data.query_label.to_numpy() == expected).all()
    assert data.loc[data.query_label.isin([ACT,REP]),'status'].eq('ok').all()
    data['TF'] = data.tf.map(map_tf)
    keys = ['source','assay','stratum','percent','method']
    assert not data.duplicated(keys+['tf']).any()
    queries, coverage, memberships, collisions = {}, [], [], []
    for key, sub in data.groupby(keys, sort=True):
        meta = dict(zip(keys,key))
        meta['query_id'] = '__'.join(map(str,key))
        meta['query_primary'] = key[2]=='aggregate'
        ok = sub[sub.status.eq('ok') & sub.TF.isin(targets)]
        universe = set(ok.TF)
        sets = {role: set(ok.loc[ok.query_label.eq(role),'TF']) for role in [ACT, REP]}
        for tf, g in ok.groupby('TF'):
            if len(g)>1:
                collisions.append(dict(**meta,TF=tf,raw_ids=';'.join(g.tf),roles=';'.join(g.query_label)))
        queries[key] = (meta, universe, sets)
        coverage.append(dict(**meta,n_source_rows=len(sub),n_status_ok=int(sub.status.eq('ok').sum()),n_estimable=len(universe),n_act=len(sets[ACT]),n_rep=len(sets[REP]),n_dual=len(sets[ACT]&sets[REP])))
        for tf in sorted(universe):
            memberships.append(dict(**meta,TF=tf,activator_like=tf in sets[ACT],repressor_like=tf in sets[REP]))
    save(mapping.values(),R/'identity_mapping_audit.tsv')
    save(collisions,R/'canonical_collisions.tsv')
    save(coverage,R/'coverage_by_condition.tsv')
    save(memberships,R/'query_memberships.tsv.gz')
    print('Queries/references:',len(queries),len(refs),flush=True)
    rows, backgrounds = [], []
    for key,(meta, estimable, sets) in queries.items():
        other = queries.get((*key[:-1], 'pwm' if key[-1]=='seq2kd' else 'seq2kd'))
        common = estimable & other[1] if other else set()
        for refmeta, eligible, refsets in refs:
            for policy in ['calculable_reference','directional_only','common_estimable']:
                u = (common if policy=='common_estimable' else estimable) & eligible
                if policy=='directional_only':
                    u &= refsets[ACT] | refsets[REP]
                base = dict(**meta,**refmeta,background_policy=policy,n_estimable=len(estimable),n_background=len(u),n_uncertain_reference_background=len(u-refsets[ACT]-refsets[REP]))
                backgrounds.append(dict(**base,TFs=';'.join(sorted(u))))
                for role, term in itertools.product([ACT,REP],repeat=2):
                    q, r = sets[role]&u, refsets[term]&u
                    h = q&r
                    a,b,c,d = len(h),len(q-r),len(r-q),len(u-(q|r))
                    status = ('empty_background' if not u else 'empty_query_set' if not q else 'empty_reference_set' if not r else 'query_equals_background' if q==u else 'reference_equals_background' if r==u else 'ok')
                    e = len(q)*len(r)/len(u) if u else np.nan
                    primary = meta['query_primary'] and refmeta['suite'] in CFG['primary_suites'] and refmeta['analysis_tier']=='main'
                    rows.append(dict(**base,query_direction=role,reference_direction=term,same_direction=role==term,n_selected=len(q),n_annotated=len(r),n_hits=len(h),a=a,b=b,c=c,d=d,expected_hits=e,fold_enrichment=len(h)/e if e>0 else np.nan,odds_ratio=a*d/(b*c) if b*c else (np.inf if a*d else np.nan),test_status=status,hits=';'.join(sorted(h)),primary_test_family=primary))
    result = pd.DataFrame(rows)
    result['p_value'] = np.nan
    valid = result.test_status.eq('ok')
    result.loc[valid,'p_value'] = hypergeom.sf(result.loc[valid,'a']-1,result.loc[valid,'n_background'],result.loc[valid,'n_annotated'],result.loc[valid,'n_selected'])
    for _,idx in result.groupby(['query_id','reference_id','background_policy']).groups.items():
        assert len(idx)==4
        result.loc[idx,'q_value_four_tests'] = bh(result.loc[idx,'p_value'])
    for policy in result.background_policy.unique():
        idx = result.index[result.primary_test_family & result.background_policy.eq(policy)]
        result.loc[idx,'q_value_primary_family'] = bh(result.loc[idx,'p_value'])
    result['q_value_all_tests'] = bh(result.p_value)
    result['minus_log10_p'] = -np.log10(result.p_value.clip(lower=1e-300))
    assert (result[['a','b','c','d']].sum(axis=1)==result.n_background).all()
    assert result.hits.map(lambda s:len(set(s.split(';'))-{''})).eq(result.n_hits).all()
    # Check Fisher independently for all focal rows, plus a deterministic sample across the full extension.
    focal = result[result.reference_id.eq('batch0__A__main__main')&result.source.eq('legacy')&result.stratum.eq('aggregate')]
    check = pd.concat([focal[focal.test_status.eq('ok')],result[valid].sample(min(512,int(valid.sum())),random_state=42)]).drop_duplicates(['query_id','reference_id','background_policy','query_direction','reference_direction'])
    for row in check.itertuples():
        p = fisher_exact([[row.a,row.b],[row.c,row.d]],alternative='greater')[1]
        np.testing.assert_allclose(p,row.p_value,rtol=1e-10,atol=1e-14)
    save(result,R/'enrichment_all_comparisons.tsv.gz')
    primary = result[result.primary_test_family]
    save(primary,R/'primary_A_B_enrichment.tsv')
    save(primary[primary.source.eq('legacy')],R/'primary_eQTL_MPRA_enrichment.tsv')
    save(focal,R/'batch0_A_main_eQTL_MPRA.tsv')
    save(backgrounds,R/'tested_background_memberships.tsv.gz')
    save(INPUTS,R/'input_manifest.tsv')
    assert all(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in INPUTS)
    validation = dict(status='PASS',planned_tests=len(result),estimable_tests=int(valid.sum()),independent_fisher_checks=len(check),input_checksums_unchanged=True,reference_sets_unchanged=True,primary_BH_includes_both_methods=True,primary_BH_separate_by_background=True)
    (R/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation),flush=True)
    return result

COLORS = {('eqtl','seq2kd'):'#2468ae',('eqtl','pwm'):'#a9bfd5',('mpra','seq2kd'):'#b66b24',('mpra','pwm'):'#ddc5af'}
SERIES = list(COLORS)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def render(data, meta, stem, limits):
    fig, axes = plt.subplots(4,1,figsize=(14,16))
    fig.subplots_adjust(left=.085,right=.94,bottom=.13,top=.90,hspace=.85)
    common = meta['background_policy']=='common_estimable'
    threshold = meta['S_threshold']
    st = f'|S| >= {threshold:g}' if pd.notna(threshold) else 'independent domain evidence'
    fig.suptitle(f"TF reference-set enrichment | batch {meta['reference_batch']} | {meta['suite']} | {st}",fontsize=16,y=.98)
    handles = [Patch(facecolor=COLORS[c],edgecolor='#333333',hatch='///' if c[1]=='pwm' else None,label=f"{'eQTL' if c[0]=='eqtl' else 'MPRA'} · {'Kd' if c[1]=='seq2kd' else 'PWM motif score'}") for c in SERIES]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=10)
    pos = np.array([i*2.6+j for i in range(4) for j in range(2)])
    for panel, ax in enumerate(axes):
        role = [ACT,REP][panel%2]
        count = panel>=2
        value = 'n_hits' if count else 'minus_log10_p'
        for j,(assay,method) in enumerate(SERIES):
            vals = []
            for pct,term in itertools.product(PCTS,[ACT,REP]):
                sub = data[data.assay.eq(assay)&data.method.eq(method)&data.percent.eq(pct)&data.query_direction.eq(role)&data.reference_direction.eq(term)]
                assert len(sub)==1
                row = sub.iloc[0]
                # Actual intersection counts are meaningful even when empty references make a test NA.
                vals.append(float(row[value]) if count or row.test_status=='ok' else np.nan)
            x = pos+(j-1.5)*.18
            bars = ax.bar(x,vals,width=.165,color=COLORS[(assay,method)],edgecolor='#333333',lw=.45,hatch='///' if method=='pwm' else None,zorder=3)
            if count:
                ax.bar_label(bars,labels=[str(int(v)) for v in vals],padding=3,fontsize=8,rotation=90)
            else:
                for xx,yy in zip(x,vals):
                    if not np.isfinite(yy):
                        ax.text(xx,.025,'NA',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7,rotation=90,color='#666666')
        if not count:
            ax.axhline(-np.log10(.05),color='#a74747',ls=(0,(4,3)),lw=.9)
            ax.text(1.005,-np.log10(.05),'P=.05',transform=ax.get_yaxis_transform(),va='bottom',fontsize=8,color='#a74747')
        ax.set_ylim(0,limits[(role,'count' if count else 'p')])
        sign = 'Positive-corr (r > 0)' if role==ACT else 'Negative-corr (r < 0)'
        ax.set_title(sign+' | regression P < .05'+(' · matching reference TFs' if count else ' · one-sided enrichment'),loc='left',fontweight='bold',fontsize=12,pad=12)
        ax.set_ylabel('Number of matching TFs' if count else '−log₁₀(enrichment P-value)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=count))
        ax.grid(axis='y',color='#e5e7eb',lw=.65,zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(pos,['Ref Activator','Ref Repressor']*4,fontsize=10)
        ax.tick_params(axis='x',length=0,pad=8)
        ax.set_xlim(-.65,pos.max()+.65)
        for i,pct in enumerate(PCTS):
            ax.text(i*2.6+.5,-.255,f'bind {pct}%',transform=ax.get_xaxis_transform(),ha='center',va='top',fontweight='bold',fontsize=11)
            if i:
                ax.axvline(i*2.6-.8,color='#d1d5db',lw=.75,zorder=0)
    bg = 'Common background: TFs estimable by both Kd and PWM, with a calculable frozen reference score.' if common else 'Method-specific background: TFs estimable by each method, with a calculable frozen reference score.'
    fig.text(.085,.025,bg+'\nAggregate effects · Raw one-sided Fisher P; dashed P=.05 is nominal, not BH-corrected significance.\nCounts = actual TF intersections (including 0); NA in P panels = no estimable test, including an empty reference set.\nSolid: Kd; pale/hatched: PWM motif score. Panel y-scales differ; paired background versions share scales.\nBinding percentages filter sites; |S| defines the frozen OE reference direction. Uncertain reference TFs remain in the background.',fontsize=9,color='#4b5563',linespacing=1.5)
    fig.savefig(D/(stem+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(D/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    save(data,D/(stem+'.source.tsv'))
    print('Saved',stem,flush=True)

def plot(result):
    legacy = result[result.source.eq('legacy')&result.stratum.eq('aggregate')&result.background_policy.isin(['calculable_reference','common_estimable'])]
    legacy = legacy[(legacy.primary_test_family)|legacy.suite.str.startswith('C-')]
    # Deliver the specifically requested plot first.
    groups = list(legacy.groupby(['reference_id','background_policy'],sort=True))
    groups.sort(key=lambda item: (item[0][0]!='batch0__A__main__main',item[0]))
    index = []
    for (rid,policy),data in groups:
        limits = {}
        paired = legacy[legacy.reference_id.eq(rid)]
        for role in [ACT,REP]:
            sub = paired[paired.query_direction.eq(role)]
            limits[(role,'p')] = max(1.6,sub.minus_log10_p.max()*1.20)
            limits[(role,'count')] = max(1,sub.n_hits.max()*1.28)
        stem = rid+'__eQTL_MPRA_with_PWM'+('__common_TFs' if policy=='common_estimable' else '')
        render(data,data.iloc[0],stem,limits)
        index.append(dict(stem=stem,reference_id=rid,background_policy=policy))
    save(index,R/'figure_index.tsv')
    return legacy

def main():
    global R, D
    parser = argparse.ArgumentParser(description='Evaluate TF overexpression reference-set overlap')
    parser.add_argument('--regressions', type=Path, required=True)
    parser.add_argument('--proteins', type=Path, required=True)
    parser.add_argument('--hgnc', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    R, D = args.output / 'results', args.output / 'figures'
    for directory in (R, D):
        directory.mkdir(parents=True, exist_ok=True)
    plot(analyze(args))


if __name__=='__main__':
    main()
