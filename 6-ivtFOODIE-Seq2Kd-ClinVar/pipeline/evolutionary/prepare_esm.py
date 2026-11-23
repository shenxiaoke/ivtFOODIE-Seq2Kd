#!/usr/bin/env python3
"""Deduplicated masked WT sequence/site tasks, no mutation position leakage."""
import pandas as pd,collections
from shared import *
def main():
 setup();c=predicted_coding();p=pd.read_parquet(RUN/'prepared/proteins.parquet').set_index('protein_id');needs=collections.defaultdict(set);seqs={};links=[]
 def add(seq,positions):
  key=digest(seq);seqs[key]=seq;needs[key].update(positions);return key
 for r in c.itertuples():
  seq=r.wt_sequence;i=r.aa_pos1-1-r.fragment_start0-r.crop_start0
  if not (0<=i<len(seq) and seq[i]==r.ref_aa):raise ValueError('DBD mutation coordinate mismatch')
  if r.consequence=='missense':
   key=add(seq,[i]);links.append(dict(mapping_id=r.mapping_id,context='dbd',sequence_key=key,site0=i,deleted_end0=-1))
   full=p.loc[r.protein_id,'full_sequence'];j=r.aa_pos1-1;start=max(0,min(j-511,len(full)-1022)) if len(full)>1022 else 0;s=full[start:start+1022]
   if s[j-start]!=r.ref_aa:raise ValueError('Full protein mutation coordinate mismatch')
   key=add(s,[j-start]);links.append(dict(mapping_id=r.mapping_id,context='full_or_1022_window',sequence_key=key,site0=j-start,deleted_end0=-1,full_window_start0=start))
  elif r.consequence=='stop_gain':
   if r.alt_sequence!=seq[:i]:raise ValueError('Truncation mismatch')
   key=add(seq,range(i,len(seq)));links.append(dict(mapping_id=r.mapping_id,context='dbd_deleted_region',sequence_key=key,site0=i,deleted_end0=len(seq)))
 parquet(pd.DataFrame(links),OUT/'tables/esm_links.parquet')
 signature=dict(esm_sha256=json.loads((RUN/'qc/inference_signature.json').read_text())['esm_sha256'],strategy='one_mask_at_a_time_WT_context_all_20_logprob_differences',precision='FP32',code_sha256=filehash(HERE/'esm_worker.py'))
 n=0
 for key in sorted(needs):
  positions=sorted(needs[key]);seq=seqs[key]
  for offset in range(0,len(positions),32):
   tid=f'{key}.{offset:04d}';task=dict(task_id=tid,sequence=seq,sequence_key=key,positions0=positions[offset:offset+32],signature=signature);task['task_hash']=digest(json.dumps(task,sort_keys=True));f=OUT/'tasks'/(tid+'.json')
   if f.exists() and json.loads(f.read_text())!=task:raise RuntimeError('Task definition changed')
   save_json(task,f);n+=1
 save_json(dict(tasks=n,sequences=len(needs),masked_sites=sum(map(len,needs.values())),missense_unique_snps=c[c.consequence=='missense'].variant_id.nunique(),stop_gain_unique_snps=c[c.consequence=='stop_gain'].variant_id.nunique(),inputs={str(q):filehash(q) for q in [RUN/'prepared/coding_variants.parquet',RUN/'prepared/proteins.parquet']},signature=signature),OUT/'qc/esm_plan.json')
 (OUT/'tasks/CLOSED').write_text(str(n));print((OUT/'qc/esm_plan.json').read_text(),flush=True)
if __name__=='__main__':main()
