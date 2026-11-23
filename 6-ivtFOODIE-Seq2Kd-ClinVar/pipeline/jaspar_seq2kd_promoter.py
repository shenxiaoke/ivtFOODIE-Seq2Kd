"""Seq2Kd on fixed promoter allele windows, with JASPAR top5 candidate selection."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from common import ROOT, filehash, save_json, parquet, digest

BASE=ROOT/'results'
BRANCH=BASE/'jaspar_seq2kd_promoter'


def prepare():
    for sub in ['prepared','qc','scores','tasks','logs','locks','cache']:
        (BRANCH/sub).mkdir(parents=True,exist_ok=True)
    links={'prepared/promoter_variants.parquet':BASE/'prepared/promoter_variants.parquet',
           'qc/inference_signature.json':BASE/'qc/inference_signature.json','embeddings':BASE/'embeddings'}
    for rel,source in links.items():
        dest=BRANCH/rel
        if not dest.exists():dest.symlink_to(source)
    proteins=pd.read_parquet(BASE/'prepared/proteins.parquet').set_index('protein_id')
    motifs=json.loads((BASE/'jaspar_coding/prepared/motifs.json').read_text())
    by_pid={m['protein_id']:m for m in motifs}
    scorefiles={}
    for f in sorted((BASE/'scores').glob('promoter.*.parquet')):
        pid=f.stem[len('promoter.'):].rsplit('.',1)[0]
        if pid in by_pid:scorefiles.setdefault(pid,[]).append(f)
    baseline=json.loads((BASE/'qc/inference_signature.json').read_text())
    annhash=filehash(BASE/'prepared/promoter_variants.parquet')
    def one(pid):
        m=by_pid[pid]
        candidate=BASE/'jaspar_promoter/candidates'/f"{m['tf']}.{m['motif_id']}.parquet"
        candidates=pd.read_parquet(candidate,columns=['variant_id'])
        ids=set(candidates.variant_id)
        candidates['protein_id']=pid;candidates['tf']=m['tf']
        parquet(candidates,BRANCH/'prepared'/f'candidates.{pid}.parquet')
        cached=[];lineage=[]
        for path in scorefiles.get(pid,[]):
            meta=json.loads(path.with_suffix('.json').read_text())
            task=json.loads((BASE/'tasks'/(path.stem+'.json')).read_text())
            assert meta['task_signature']==task['signature']
            assert task['sequence']==proteins.loc[pid,'promoter_sequence']
            assert task['prepared_inputs']['promoter_variants.parquet']==annhash
            assert all(meta['inference'][k]==v for k,v in baseline.items())
            frame=pd.read_parquet(path)
            assert set(frame.variant_id)==set(task['variant_ids'])
            frame=frame[frame.variant_id.isin(ids)]
            if len(frame):
                cached.append(frame)
                lineage.append(dict(path=str(path),score_sha256=filehash(path),metadata_sha256=filehash(path.with_suffix('.json'))))
        if cached:
            cache=pd.concat(cached,ignore_index=True)
            assert not cache.variant_id.duplicated().any()
            assert cache.protein_id.eq(pid).all()
            parquet(cache,BRANCH/'cache'/f'{pid}.parquet')
            ids-=set(cache.variant_id)
        remaining=sorted(ids)
        for start in range(0,len(remaining),2048):
            task=dict(task_id=f'promoter.{pid}.{start:05d}',stage='promoter',protein_id=pid,tf=m['tf'],
                sequence=proteins.loc[pid,'promoter_sequence'],variant_ids=remaining[start:start+2048],
                prepared_inputs={'promoter_variants.parquet':annhash},candidate_files={str(candidate):filehash(candidate)})
            task['signature']=digest(json.dumps(task,sort_keys=True))
            save_json(task,BRANCH/'tasks'/(task['task_id']+'.json'))
        return dict(protein_id=pid,candidates=len(candidates),cached=len(candidates)-len(remaining),new=len(remaining),lineage=lineage)
    records=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(one,sorted(by_pid)):
            records.append(record)
            if len(records)%50==0:print(f'Prepared {len(records)}/{len(by_pid)} fragments',flush=True)
    save_json(records,BRANCH/'qc/cache_lineage.json')
    summary={key:sum(r[key] for r in records) for key in ['candidates','cached','new']}
    summary['tasks']=len(list((BRANCH/'tasks').glob('*.json')))
    summary['definition']='JASPAR human CORE exact TF selection as existing coding branch; top5 PWM binding candidates over all promoter SNVs; fixed observed 40bp REF/ALT windows, same Seq2Kd protein sequences; max |delta pKd| over fragments within TF, then mean over TFs per unique SNV.'
    save_json(summary,BRANCH/'qc/preparation.json');save_json({'queue_closed':True},BRANCH/'tasks/CLOSED')
    print(json.dumps(summary,indent=2),flush=True)


def render():
    import analyze as original
    candidates=pd.concat([pd.read_parquet(p) for p in sorted((BRANCH/'prepared').glob('candidates.*.parquet'))],ignore_index=True)
    frames=[pd.read_parquet(p) for p in sorted((BRANCH/'cache').glob('*.parquet'))]
    baseline=json.loads((BASE/'qc/inference_signature.json').read_text())
    tasks=list((BRANCH/'tasks').glob('*.json'))
    for path in tasks:
        task=json.loads(path.read_text());score=BRANCH/'scores'/(path.stem+'.parquet')
        meta=json.loads(score.with_suffix('.json').read_text())
        assert meta['task_signature']==task['signature']
        assert all(meta['inference'][k]==v for k,v in baseline.items())
        d=pd.read_parquet(score)
        assert set(d.variant_id)==set(task['variant_ids']) and d.protein_id.eq(task['protein_id']).all()
        frames.append(d)
    scores=pd.concat(frames,ignore_index=True)
    assert not scores.duplicated(['variant_id','protein_id']).any()
    pair=candidates.merge(scores.drop(columns='tf'),on=['variant_id','protein_id'],validate='one_to_one')
    assert len(pair)==len(candidates)==len(scores)
    assert np.isfinite(pair[['ref_pkd','alt_pkd']]).all().all()
    pair['delta']=pair.alt_pkd-pair.ref_pkd;pair['abs_delta']=pair.delta.abs()
    parquet(pair,BRANCH/'analysis/promoter_pair_scores/seq2kd_jaspar.parquet')
    tf=pair.groupby(['variant_id','tf'],as_index=False).agg(value=('abs_delta','max'))
    d=tf.groupby('variant_id',as_index=False).agg(mean=('value','mean'),n_tf=('tf','size'))
    ann=pd.read_parquet(BASE/'prepared/promoter_variants.parquet')
    d=d.merge(ann[['variant_id','label','review_stars']],validate='one_to_one')
    endpoint='promoter.seq2kd_jaspar.top5.mean'
    original.OUT=BRANCH/'analysis';original.FIG=BASE/'analysis/figures'
    original.compare(d,endpoint,'mean','|delta pKd|','Promoter / seq2kd_jaspar / top 5% / mean across TFs',primary=False)
    pd.DataFrame(original.TESTS).to_csv(BRANCH/'analysis/statistical_tests_unadjusted.csv',index=False)
    parquet(d,BASE/'analysis/variant_summaries'/f'{endpoint}.parquet')
    save_json(dict(complete=True,candidate_pairs=len(pair),unique_snvs=len(d),tasks=len(tasks)),BRANCH/'COMPLETE.json')
    print(pd.DataFrame(original.TESTS).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','render']);args=parser.parse_args()
    globals()[args.stage]()
