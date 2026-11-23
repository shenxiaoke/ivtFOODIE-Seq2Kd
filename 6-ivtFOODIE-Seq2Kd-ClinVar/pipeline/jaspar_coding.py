#!/usr/bin/env python3
"""Add a provenance-tracked JASPAR coding branch without changing existing analyses."""
import argparse, json, re, os, html
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from common import ROOT, RUN, filehash, save_json, parquet, digest, LABELS
from motifs import parse_meme, sample_dna

BRANCH = ROOT / 'results/jaspar_coding'
META_URL = 'https://mencius-pub.uio.no/JASPAR/JASPAR_metadata/2026/ultimate_metadata_table_CORE.tsv'
MEME_URL = 'https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt'

def prepare():
    for sub in ['inputs','prepared','qc','tasks','scores','locks','logs']:
        (BRANCH/sub).mkdir(parents=True, exist_ok=True)
    urls={'metadata.tsv': META_URL, 'JASPAR2026_CORE_vertebrates_non-redundant.meme': MEME_URL}
    for name,url in urls.items():
        dest=BRANCH/'inputs'/name
        if not dest.exists():
            response=requests.get(url,timeout=120); response.raise_for_status()
            dest.write_bytes(response.content)
    metadata=pd.read_csv(BRANCH/'inputs/metadata.tsv',sep='\t',dtype=str).fillna('')
    motifs=parse_meme(BRANCH/'inputs/JASPAR2026_CORE_vertebrates_non-redundant.meme')
    by_id=metadata.set_index('matrix_id').to_dict('index')
    eligible=[]
    for m in motifs:
        r=by_id[m['motif_id']]
        if r['collection']!='CORE' or r['tax_group']!='vertebrates': continue
        if '9606' not in re.findall(r'\d+',r['tax_id']): continue
        # Composite motifs cannot be assigned to a single mutated protein.
        if '::' in r['name'] or len(m['probabilities'])>40: continue
        eligible.append((m,r))
    proteins=pd.read_parquet(ROOT/'results/prepared/proteins.parquet')
    coding=pd.read_parquet(ROOT/'results/prepared/coding_variants.parquet')
    coding=coding[coding.status.eq('ready')]
    selections=[]; selected=[]; samples=[]
    for p in proteins.itertuples():
        candidates=[]
        for m,r in eligible:
            ids=re.findall(r'[A-Z0-9]+(?:-\d+)?',r['uniprot_ids'])
            accession=p.uniprot.split('.')[0]
            match_accession=accession in ids
            match_symbol=r['name']==p.tf
            if match_accession or match_symbol:
                candidates.append((0 if match_accession else 1,int(r['base_id'][2:]),-int(r['version']),m,r))
        candidates.sort(key=lambda x:x[:3])
        if not candidates:
            selections.append(dict(protein_id=p.protein_id,tf=p.tf,status='no_exact_human_single_factor_motif'))
            continue
        priority,_,_,m,r=candidates[0]
        m=dict(m,source='jaspar',protein_id=p.protein_id,tf=p.tf)
        selected.append(m)
        selections.append(dict(protein_id=p.protein_id,tf=p.tf,status='selected',motif_id=m['motif_id'],
            motif_name=r['name'],match='UniProt' if priority==0 else 'exact_gene_symbol',tax_id=r['tax_id'],
            candidates=';'.join(c[3]['motif_id'] for c in candidates)))
        for i,dna in enumerate(sample_dna(m)):
            samples.append(dict(protein_id=p.protein_id,source='jaspar',sample_index=i,dna_seq=dna))
    parquet(pd.DataFrame(samples),BRANCH/'prepared/sampled_dna.parquet')
    save_json(selected,BRANCH/'prepared/motifs.json')
    pd.DataFrame(selections).to_csv(BRANCH/'qc/motif_mapping.csv',index=False)
    covered={m['protein_id'] for m in selected}
    use=coding[coding.protein_id.isin(covered)]
    parquet(use,BRANCH/'prepared/coding_variants.parquet')
    signature={str(p):filehash(p) for p in [ROOT/'results/prepared/coding_variants.parquet',BRANCH/'prepared/sampled_dna.parquet',BRANCH/'prepared/motifs.json']}
    for pid,g in use.groupby('protein_id',sort=True):
        for start in range(0,len(g),8):
            records=[]
            for r in g.iloc[start:start+8].to_dict('records'):
                records.append({k:r[k] for k in ['mapping_id','variant_id','protein_id','tf','wt_sequence','alt_sequence','consequence']}|{'sources':['jaspar']})
            task=dict(task_id=f'coding.jaspar.{pid}.{start:05d}',stage='coding',records=records,prepared_inputs=signature)
            task['signature']=digest(json.dumps(task,sort_keys=True))
            save_json(task,BRANCH/'tasks'/(task['task_id']+'.json'))
    for dest,source in [(BRANCH/'embeddings',ROOT/'results/embeddings'),(BRANCH/'qc/inference_signature.json',ROOT/'results/qc/inference_signature.json')]:
        if not dest.exists(): dest.symlink_to(source)
    save_json(dict(queue_closed=True),BRANCH/'tasks/CLOSED')
    summary=dict(release='JASPAR2026',collection='CORE',human_tax_id=9606,nonredundant=True,
        selection_rule='Single-factor only; exact UniProt first, then exact gene symbol; smallest numeric base ID, latest version in nonredundant file; no homolog substitution.',
        dna_sampling='100 fixed 40bp samples; centered motif, uniform background; existing SHA256 seed42 scheme; paired WT/ALT DNA',
        eligible_human_motifs=len(eligible),covered_fragments=len(selected),coding_mapping_rows=len(use),
        unique_snvs=int(use.variant_id.nunique()),consequence_counts=use.groupby('consequence').variant_id.nunique().to_dict(),
        tasks=len(list((BRANCH/'tasks').glob('*.json'))),sources={name:dict(url=url,sha256=filehash(BRANCH/'inputs'/name)) for name,url in urls.items()},
        prepared_inputs=signature)
    save_json(summary,BRANCH/'qc/preparation.json');print(json.dumps(summary,indent=2),flush=True)

def render():
    import analyze as original
    out=ROOT/'results/analysis'; branchout=BRANCH/'analysis'
    tasks=[json.loads(p.read_text()) for p in sorted((BRANCH/'tasks').glob('*.json'))]
    assert tasks and all((BRANCH/'scores'/(t['task_id']+'.json')).exists() for t in tasks)
    frames=[]; inference_signatures=[]
    for t in tasks:
        p=BRANCH/'scores'/(t['task_id']+'.parquet')
        meta=json.loads(p.with_suffix('.json').read_text())
        assert meta['task_signature']==t['signature']
        inference_signatures.append(meta['inference'])
        frame=pd.read_parquet(p);raw=pd.read_parquet(p.with_name(p.stem+'.raw.parquet'))
        assert set(frame.mapping_id)=={r['mapping_id'] for r in t['records']}
        for r in raw.itertuples():
            ref=np.array(r.ref_pkd,dtype=np.float32);alt=np.array(r.alt_pkd,dtype=np.float32)
            assert len(ref)==len(alt)==100 and np.isfinite(ref).all() and np.isfinite(alt).all()
            score=frame[(frame.mapping_id==r.mapping_id)&frame.n_dna.eq(100)].mean_abs_delta.iloc[0]
            assert float(np.abs(alt-ref).mean())==score
        frames.append(frame)
    assert all(s==inference_signatures[0] for s in inference_signatures)
    scores=pd.concat(frames,ignore_index=True);assert not scores.duplicated(['mapping_id','n_dna']).any()
    coding=pd.read_parquet(BRANCH/'prepared/coding_variants.parquet')
    assert set(scores.mapping_id)==set(coding.mapping_id)
    use=scores[scores.n_dna.eq(100)].merge(coding[['mapping_id','label','consequence','review_stars']],on='mapping_id',validate='one_to_one')
    data=use.groupby('variant_id',as_index=False).agg(value=('mean_abs_delta','mean'),label=('label','first'))
    assert not data.variant_id.duplicated().any()
    parquet(use,branchout/'mapping_summaries.parquet')
    endpoint='coding.jaspar.all.n100'
    # Keep new statistics separate: do not alter the existing FDR family.
    original.OUT=branchout;original.FIG=out/'figures'
    original.compare(data,endpoint,'value','Mean |delta pKd| across sampled DNA','Coding / all / JASPAR DNA / n=100',primary=False)
    pd.DataFrame(original.TESTS).to_csv(branchout/'statistical_tests_unadjusted.csv',index=False)
    pd.DataFrame(original.SUMMARIES).to_csv(branchout/'group_distributions.csv',index=False)
    files = []
    for source in ['streme', 'jaspar']:
        for ext in ['png', 'pdf', 'svg']:
            path = out/'figures'/f'coding.{source}.all.n100.{ext}'
            files.append(dict(path=str(path), sha256=filehash(path)))
    frozen=json.loads((BRANCH/'qc/original_files_before.json').read_text())
    assert all(filehash(p)==h for p,h in frozen.items()),'An original output changed'
    save_json(dict(complete=True,unique_snvs=len(data),mapping_rows=len(use),label_counts=data.label.value_counts().to_dict(),
        raw_predictions_verified=True,inference_signature=inference_signatures[0],original_files_unchanged=True,
        task_count=len(tasks),files=files,script_sha256=filehash(__file__)),BRANCH/'COMPLETE.json')
    page=['<!doctype html><meta charset="utf-8"><title>Coding motif sources</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}img{max-width:100%}</style><h1>Coding: STREME / JASPAR</h1><p>All coding variants; 100 paired DNA sequences. Each source uses its available SNV cohort; these are not matched cohorts.</p>']
    for source in ['streme','jaspar']:
        for directory,label in [('figures','symlog')]:
            name=f'coding.{source}.all.n100'
            page.append(f'<h2>{source.upper()} — {label}</h2><a href="{directory}/{name}.pdf">PDF</a> · <a href="{directory}/{name}.svg">SVG</a><br><img src="{directory}/{name}.png" alt="{source} {label}">')
    (out/'index_coding_three_sources.html').write_text('\n'.join(page))
    (BRANCH/'README.md').write_text('# JASPAR coding supplement\n\nJASPAR2026 CORE nonredundant human single-factor motifs.\n\nSee `qc/preparation.json` for sources, checksums and selection rules; `qc/motif_mapping.csv` for every fragment and candidate motif.\n\n`analysis/variant_summaries/coding.jaspar.all.n100.parquet` contains one observation per SNV. `scores/*.raw.parquet` retains all 100 paired WT/ALT predictions.\n\nFigures: `../analysis/figures/coding.jaspar.all.n100.{png,pdf,svg}`. Existing STREME plots were reused unchanged.\n\nThe two sources have different available cohorts. JASPAR summary statistics are separate and unadjusted; the original analysis FDR family was not refitted.\n')
    print(json.dumps(dict(complete=True,n=len(data),labels=data.label.value_counts().to_dict())),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render']);a=p.parse_args()
    prepare() if a.stage=='prepare' else render()
