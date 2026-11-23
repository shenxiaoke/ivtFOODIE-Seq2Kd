#!/usr/bin/env python3
"""Extend JASPAR Seq2Kd to nested 1/5/10/30% thresholds in an isolated run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, digest, filehash, parquet, save_json

BASE = ROOT/'results'
RUN = BASE/'jaspar_seq2kd_promoter_all_thresholds'
PCTS = [1,5,10,30]
V = REF = ALT = None


def scan(m):
    from candidates import score_matrix
    from motifs import pwm_scores
    path = RUN/'candidates'/f"{m['tf']}.{m['motif_id']}.parquet"
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        return json.loads(meta.read_text())
    mat = np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
    ref = score_matrix(REF,mat); alt = score_matrix(ALT,mat)
    assert np.allclose(ref[:64],pwm_scores(REF[:64],m),atol=1e-10)
    assert np.allclose(alt[:64],pwm_scores(ALT[:64],m),atol=1e-10)
    binding = np.maximum(ref,alt)
    cuts = {pct:float(np.percentile(binding,100-pct)) for pct in PCTS}
    keep = binding >= cuts[30]
    d = V.loc[keep,['variant_id']].copy()
    d['tf'] = m['tf']; d['motif_id'] = m['motif_id']
    for pct in PCTS:
        d[f'top{pct}'] = binding[keep] >= cuts[pct]
    old = pd.read_parquet(BASE/'jaspar_promoter/candidates'/path.name,columns=['variant_id'])
    assert set(d.loc[d.top5,'variant_id']) == set(old.variant_id)
    parquet(d,path)
    record = dict(tf=m['tf'],motif_id=m['motif_id'],cutoffs=cuts,
                  counts={pct:int(d[f'top{pct}'].sum()) for pct in PCTS},
                  path=str(path),sha256=filehash(path),original_top5_reproduced=True)
    save_json(record,meta)
    return record


def prepare(workers):
    global V,REF,ALT
    from motifs import encode
    from candidates import score_matrix
    for sub in ['candidates','prepared','cache','tasks','scores','logs','locks','qc','analysis/variant_summaries']:
        (RUN/sub).mkdir(parents=True,exist_ok=True)
    for dest,source in [('prepared/promoter_variants.parquet',BASE/'prepared/promoter_variants.parquet'),
                        ('qc/inference_signature.json',BASE/'qc/inference_signature.json'),
                        ('embeddings',BASE/'embeddings')]:
        target = RUN/dest
        if not target.exists(): target.symlink_to(source)
    V = pd.read_parquet(BASE/'prepared/promoter_variants.parquet')
    assert not V.variant_id.duplicated().any()
    REF = encode(V.ref_seq); ALT = encode(V.alt_seq)
    motifpath = BASE/'jaspar_coding/prepared/motifs.json'
    motifs = json.loads(motifpath.read_text())
    unique = {(m['tf'],m['motif_id']):m for m in motifs}
    frozen = {str(p):filehash(p) for p in (BASE/'analysis/requested_median_pvalue_figures').glob('*.pdf')}
    save_json(frozen,RUN/'qc/original_pdf_hashes.json')
    save_json(dict(variants_sha256=filehash(BASE/'prepared/promoter_variants.parquet'),
                   motifs_sha256=filehash(motifpath),universe=len(V),unique_motifs=len(unique),
                   fragment_assignments=len(motifs),pcts=PCTS,
                   rule='Per motif max(REF,ALT) double-strand full-window PWM score; percentile over all promoter SNVs; ties included; no clinical labels in selection.'),RUN/'qc/inputs.json')
    references = [BASE/'prepared/promoter_variants.parquet',motifpath,
                  BASE/'jaspar_seq2kd_promoter/analysis/promoter_pair_scores/seq2kd_jaspar.parquet',
                  BASE/'analysis/variant_summaries/promoter.seq2kd_jaspar.top5.mean.parquet',
                  BASE/'analysis/requested_median_pvalue_figures/source_statistics.csv']
    save_json({str(p):filehash(p) for p in references},RUN/'qc/source_reference_hashes.json')
    score_matrix(REF[:1],np.ones((1,4)))
    print(f'Scanning {len(unique)} motifs over {len(V)} SNVs',flush=True)
    records=[]
    with mp.get_context('fork').Pool(workers) as pool:
        for r in pool.imap_unordered(scan,[unique[k] for k in sorted(unique)]):
            records.append(r)
            if len(records)%25 == 0: print(f'Scanned {len(records)}/{len(unique)}',flush=True)
    save_json(records,RUN/'qc/thresholds.json')
    del REF,ALT
    by_pid={m['protein_id']:m for m in motifs}
    proteins=pd.read_parquet(BASE/'prepared/proteins.parquet').set_index('protein_id')
    signature=json.loads((BASE/'qc/inference_signature.json').read_text())
    annhash=filehash(BASE/'prepared/promoter_variants.parquet')
    top5=pd.read_parquet(BASE/'jaspar_seq2kd_promoter/analysis/promoter_pair_scores/seq2kd_jaspar.parquet')
    top5_by={pid:g[['variant_id','protein_id','tf','ref_pkd','alt_pkd']].copy()
             for pid,g in top5.groupby('protein_id',sort=False)}
    del top5
    basefiles={}
    for path in sorted((BASE/'scores').glob('promoter.*.parquet')):
        pid=path.stem[len('promoter.'):].rsplit('.',1)[0]
        if pid in by_pid:basefiles.setdefault(pid,[]).append(path)
    complete=json.loads((BASE/'jaspar_seq2kd_promoter/COMPLETE.json').read_text())
    assert complete['complete']
    def one(pid):
        ready=RUN/'qc'/f'prepared.{pid}.json'
        if ready.exists():return json.loads(ready.read_text())
        m=by_pid[pid]
        candidate=RUN/'candidates'/f"{m['tf']}.{m['motif_id']}.parquet"
        d=pd.read_parquet(candidate)
        d['protein_id']=pid
        parquet(d,RUN/'prepared'/f'candidates.{pid}.parquet')
        ids=set(d.variant_id)
        # Preserve the exact top5 values already independently verified and used
        # in published figures. Its branch has the same model/input signatures.
        cached=[top5_by[pid]] if pid in top5_by else []
        if cached:ids -= set(cached[0].variant_id)
        lineage=[]
        for path in basefiles.get(pid,[]):
            meta=json.loads(path.with_suffix('.json').read_text())
            task=json.loads((BASE/'tasks'/(path.stem+'.json')).read_text())
            assert meta['task_signature']==task['signature']
            assert task['sequence']==proteins.loc[pid,'promoter_sequence']
            assert task['prepared_inputs']['promoter_variants.parquet']==annhash
            assert all(meta['inference'][k]==v for k,v in signature.items())
            frame=pd.read_parquet(path,columns=['variant_id','protein_id','tf','ref_pkd','alt_pkd'])
            assert set(frame.variant_id)==set(task['variant_ids'])
            assert frame.protein_id.eq(pid).all()
            frame=frame[frame.variant_id.isin(ids)]
            if len(frame):
                cached.append(frame);ids-=set(frame.variant_id)
                lineage.append(dict(path=str(path),sha256=filehash(path),task_signature=task['signature']))
        if cached:
            cache=pd.concat(cached,ignore_index=True)
            assert not cache.variant_id.duplicated().any()
            assert np.isfinite(cache[['ref_pkd','alt_pkd']]).all().all()
            parquet(cache,RUN/'cache'/f'{pid}.parquet')
        remaining=sorted(ids)
        chash=filehash(candidate)
        for start in range(0,len(remaining),2048):
            task=dict(task_id=f'promoter.{pid}.{start:05d}',stage='promoter',protein_id=pid,tf=m['tf'],
                      sequence=proteins.loc[pid,'promoter_sequence'],variant_ids=remaining[start:start+2048],
                      prepared_inputs={'promoter_variants.parquet':annhash},candidate_files={str(candidate):chash})
            task['signature']=digest(json.dumps(task,sort_keys=True))
            save_json(task,RUN/'tasks'/(task['task_id']+'.json'))
        r=dict(protein_id=pid,candidates=len(d),cached=len(d)-len(remaining),new=len(remaining),lineage=lineage)
        save_json(r,ready)
        return r
    results=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in pool.map(one,sorted(by_pid)):
            results.append(r)
            if len(results)%50==0:print(f'Prepared {len(results)}/{len(by_pid)} fragments',flush=True)
    save_json(results,RUN/'qc/cache_lineage.json')
    summary={k:sum(r[k] for r in results) for k in ['candidates','cached','new']}
    summary['tasks']=len(list((RUN/'tasks').glob('*.json')))
    save_json(summary,RUN/'qc/preparation.json')
    save_json({'queue_closed':True},RUN/'tasks/CLOSED')
    print(json.dumps(summary,indent=2),flush=True)


def render():
    from scipy import stats
    from sklearn.metrics import roc_auc_score,average_precision_score
    import plot_requested_median_pvalues as plot
    sig=json.loads((BASE/'qc/inference_signature.json').read_text())
    ann=pd.read_parquet(BASE/'prepared/promoter_variants.parquet')[['variant_id','label','review_stars']]
    summaries={pct:[] for pct in PCTS}
    missing=[]
    for taskpath in sorted((RUN/'tasks').glob('*.json')):
        if not (RUN/'scores'/taskpath.name).exists():missing.append(taskpath.name)
    if missing:raise RuntimeError(f'{len(missing)} inference tasks not finished')
    taskmap={}
    for tp in sorted((RUN/'tasks').glob('*.json')):
        t=json.loads(tp.read_text());taskmap.setdefault(t['protein_id'],[]).append((tp,t))
    files=sorted((RUN/'prepared').glob('candidates.*.parquet'))
    # Per fragment, retain candidates then combine fragments with max per TF.
    per_tf={}
    for i,path in enumerate(files):
        d=pd.read_parquet(path);pid=d.protein_id.iloc[0];tf=d.tf.iloc[0]
        pieces=[]
        cached=RUN/'cache'/f'{pid}.parquet'
        if cached.exists():pieces.append(pd.read_parquet(cached))
        for tp,t in taskmap.get(pid,[]):
            p=RUN/'scores'/(tp.stem+'.parquet')
            meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['task_signature']==t['signature']
            assert all(meta['inference'][k]==v for k,v in sig.items())
            f=pd.read_parquet(p);assert set(f.variant_id)==set(t['variant_ids'])
            pieces.append(f[['variant_id','protein_id','tf','ref_pkd','alt_pkd']])
        s=pd.concat(pieces,ignore_index=True)
        assert not s.duplicated(['variant_id','protein_id']).any()
        pair=d.merge(s[['variant_id','protein_id','ref_pkd','alt_pkd']],on=['variant_id','protein_id'],validate='one_to_one')
        assert len(pair)==len(d)==len(s)
        assert np.isfinite(pair[['ref_pkd','alt_pkd']]).all().all()
        pair['abs_delta']=(pair.alt_pkd-pair.ref_pkd).abs()
        # Store raw prediction pair shards in the new branch for reproducibility.
        parquet(pair,RUN/'analysis/pair_scores'/f'{pid}.parquet')
        per_tf.setdefault(tf,[]).append(pair[['variant_id','abs_delta']+[f'top{p}' for p in PCTS]])
        if (i+1)%100==0:print(f'Validated {i+1}/{len(files)} prediction fragments',flush=True)
    for tf,parts in per_tf.items():
        pair=pd.concat(parts,ignore_index=True)
        for pct in PCTS:
            g=pair[pair[f'top{pct}']].groupby('variant_id',as_index=False).abs_delta.max()
            g['tf']=tf;summaries[pct].append(g)
    rows=[]
    for pct in PCTS:
        tf=pd.concat(summaries[pct],ignore_index=True)
        assert not tf.duplicated(['variant_id','tf']).any()
        d=tf.groupby('variant_id',as_index=False).agg(mean=('abs_delta','mean'),n_tf=('tf','size'))
        d=d.merge(ann,on='variant_id',validate='one_to_one')
        parquet(d,RUN/'analysis/variant_summaries'/f'promoter.seq2kd_jaspar.top{pct}.mean.parquet')
        if pct==5:
            original=pd.read_parquet(BASE/'analysis/variant_summaries/promoter.seq2kd_jaspar.top5.mean.parquet').set_index('variant_id')
            check=d.set_index('variant_id').loc[original.index]
            assert len(d)==len(original)
            assert (check.label==original.label).all()
            assert np.allclose(check['mean'],original['mean'],rtol=1e-7,atol=1e-8)
        b=d.loc[d.label.eq('Benign'),'mean'].to_numpy(dtype=float)
        p=d.loc[d.label.eq('Pathogenic'),'mean'].to_numpy(dtype=float)
        y=d.label.isin(['Benign','Pathogenic']);binary=d.loc[y,'label'].eq('Pathogenic').astype(int)
        auc=float(roc_auc_score(binary,d.loc[y,'mean']))
        test=stats.mannwhitneyu(p,b,alternative='two-sided',method='auto')
        assert np.isclose(auc,test.statistic/(len(p)*len(b)),atol=1e-12,rtol=0)
        rows.append(dict(endpoint=f'promoter.seq2kd_jaspar.top{pct}.mean',label=f'JASPAR\ntop {pct}%',
                         benign_n=len(b),pathogenic_n=len(p),benign_median=float(np.median(b)),
                         pathogenic_median=float(np.median(p)),median_difference=float(np.median(p)-np.median(b)),
                         raw_p=float(test.pvalue),auc_pathogenic_higher=auc,
                         cliffs_delta_pathogenic_minus_benign=2*auc-1,
                         auprc_average_precision=float(average_precision_score(binary,d.loc[y,'mean'])),
                         pathogenic_prevalence_pr_baseline=float(binary.mean())))
    jaspar=pd.DataFrame(rows)
    jaspar.to_csv(RUN/'analysis/jaspar_threshold_statistics.csv',index=False)
    src=pd.read_csv(plot.OUT/'source_statistics.csv').set_index('endpoint')
    aucs=pd.read_csv(plot.OUT/'02_promoter_motif_threshold_auc_cliffs_delta.csv').set_index('endpoint')
    streme=src.loc[[f'promoter.seq2kd_streme.top{p}.mean' for p in PCTS]].reset_index()
    streme['auc_pathogenic_higher']=aucs.loc[streme.endpoint,'auc_pathogenic_higher'].to_numpy()
    combined=pd.concat([streme,jaspar],ignore_index=True)
    combined.to_csv(RUN/'analysis/streme_jaspar_all_thresholds_plot_data.csv',index=False)
    name='02_promoter_streme_jaspar_all_thresholds_auc.pdf'
    plot.paired_bars(combined,'Promoter variants: STREME and JASPAR binding thresholds (mean)',name,17.5,use_auc=True)
    frozen=json.loads((RUN/'qc/original_pdf_hashes.json').read_text())
    assert frozen=={p:filehash(p) for p in frozen}
    references=json.loads((RUN/'qc/source_reference_hashes.json').read_text())
    assert references=={p:filehash(ROOT.parent/p) for p in references}
    save_json(dict(complete=True,all_original_pdf_hashes_unchanged=True,
                   reference_input_hashes_unchanged=True,
                   top5_reproduced=True,unique_snv_checks=True,all_prediction_pairs_complete=True,
                   auc_limits=[.5,plot.auc_upper(combined.auc_pathogenic_higher)],
                   output_pdf=str(plot.OUT/name),script_sha256=filehash(__file__),
                   plotting_script_sha256=filehash(plot.__file__)),RUN/'COMPLETE.json')
    print(jaspar.to_string(index=False),flush=True)
    print('NEW PDF: '+str(plot.OUT/name),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render']);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();prepare(a.workers) if a.stage=='prepare' else render()
