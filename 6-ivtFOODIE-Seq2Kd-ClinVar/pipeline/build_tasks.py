#!/usr/bin/env python3
"""Immutable task descriptions; paired scores are keyed by variant, fragment and source."""
import argparse,json
import pandas as pd
from common import *
def put(task):
 key=task['task_id'];task['prepared_inputs']=INPUT_SIGNATURE;task['signature']=digest(json.dumps(task,sort_keys=True))
 dest=RUN/'tasks'/f'{key}.json'
 if dest.exists() and json.loads(dest.read_text())!=task:raise ValueError('Immutable task changed: '+key)
 save_json(task,dest)

INPUT_SIGNATURE={}
def main():
 global INPUT_SIGNATURE
 INPUT_SIGNATURE={name:filehash(RUN/'prepared'/name) for name in ['coding_variants.parquet','promoter_variants.parquet','sampled_dna.parquet','motifs.json']}
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['coding','promoter']);args=a.parse_args()
 if args.stage=='coding':
  coding=pd.read_parquet(RUN/'prepared/coding_variants.parquet');coding=coding[coding.status=='ready']
  motifs=json.loads((RUN/'prepared/motifs.json').read_text());avail={(m['source'],m['protein_id']) for m in motifs};n=0
  for pid,g in coding.groupby('protein_id',sort=True):
   for start in range(0,len(g),8):
    chunk=g.iloc[start:start+8];records=[]
    for r in chunk.to_dict('records'):
     sources=[s for s in ['streme'] if (s,pid) in avail]
     if sources:records.append({k:r[k] for k in ['mapping_id','variant_id','protein_id','tf','wt_sequence','alt_sequence','consequence']}|{'sources':sources})
    if records:put(dict(task_id=f'coding.{pid}.{start:05d}',stage='coding',records=records));n+=1
  save_json({'coding_tasks':n,'coding_mappings_with_motifs':sum(any((s,pid) in avail for s in ['streme']) for pid in coding.protein_id)},RUN/'qc/coding_tasks.json')
 else:
  proteins=pd.read_parquet(RUN/'prepared/proteins.parquet').set_index('protein_id');n=0
  for pid,r in proteins.iterrows():
   files=[RUN/f'candidates/{s}/{pid}.parquet' for s in ['streme']]
   files=[f for f in files if f.exists()]
   if not files:continue
   ids=sorted(set().union(*(set(pd.read_parquet(f,columns=['variant_id']).variant_id) for f in files)))
   imported_path=RUN/'qc/imported_promoter_pairs.parquet'
   if imported_path.exists():
    if not hasattr(main,'_imported_pairs'):main._imported_pairs=pd.read_parquet(imported_path).groupby('protein_id').variant_id.agg(set).to_dict()
    ids=[vid for vid in ids if vid not in main._imported_pairs.get(pid,set())]
   # Chunks allow long runs to resume with at most 2048 pairs lost on interruption.
   for start in range(0,len(ids),2048):
    put(dict(task_id=f'promoter.{pid}.{start:05d}',stage='promoter',protein_id=pid,tf=r.tf,sequence=r.promoter_sequence,
      variant_ids=ids[start:start+2048],candidate_files={str(f):filehash(f) for f in files}));n+=1
  save_json({'promoter_tasks':n},RUN/'qc/promoter_tasks.json');save_json({'queue_closed':True},RUN/'tasks/CLOSED')
 print('Tasks',args.stage,n,flush=True)
if __name__=='__main__':main()
