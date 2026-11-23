#!/usr/bin/env python3
"""Fail closed on any incomplete task, changed provenance, duplicate or truncated pair."""
import json
import numpy as np,pandas as pd
from common import *
def verify(stage_filter=None):
 if not (RUN/'tasks/CLOSED').exists() and stage_filter is None:raise RuntimeError('Task queue not finalized')
 imported_manifest=RUN/'qc/cache_import_manifest.json'
 if imported_manifest.exists():
  im=json.loads(imported_manifest.read_text());src=Path(im['source_run'])
  if filehash(src/'prepared/promoter_variants.parquet')!=im['source_promoter_sha256'] or filehash(RUN/'prepared/promoter_variants.parquet')!=im['target_promoter_sha256']:raise RuntimeError('Imported DNA provenance changed')
  if filehash(src/'prepared/proteins.parquet')!=im['source_proteins_sha256'] or filehash(RUN/'prepared/proteins.parquet')!=im['target_proteins_sha256']:raise RuntimeError('Imported protein provenance changed')
  ov=pd.read_parquet(src/'prepared/promoter_variants.parquet').set_index('variant_id');nv=pd.read_parquet(RUN/'prepared/promoter_variants.parquet').set_index('variant_id');common=ov.index.intersection(nv.index)
  if not ov.loc[common,['ref_seq','alt_seq']].equals(nv.loc[common,['ref_seq','alt_seq']]):raise RuntimeError('Imported allele sequences differ')
  if filehash(RUN/'prepared/sampled_dna.parquet')!=im['sampled_dna_sha256'] or filehash(src/'prepared/sampled_dna.parquet')!=im['sampled_dna_sha256']:raise RuntimeError('Imported sampled DNA changed')
 expected=json.loads((RUN/'qc/inference_signature.json').read_text());inputs={name:filehash(RUN/'prepared'/name) for name in ['coding_variants.parquet','promoter_variants.parquet','sampled_dna.parquet','motifs.json']}
 tasks=[p for p in (RUN/'tasks').glob('*.json') if stage_filter is None or p.stem.startswith(stage_filter+'.')];counts={'coding':0,'promoter':0};rows={'coding':0,'promoter':0}
 for path in tasks:
  task=json.loads(path.read_text());target=RUN/'scores'/(path.stem+'.parquet');meta=target.with_suffix('.json')
  if not target.exists() or not meta.exists():raise RuntimeError('Missing output '+path.stem)
  if task['prepared_inputs']!=inputs:raise RuntimeError('Prepared inputs changed '+path.stem)
  actual=json.loads(meta.read_text());runtime=dict(actual.get('inference',{}))
  backend=runtime.pop('runtime_backend',None);runtime_sha=runtime.pop('runtime_source_sha256',None)
  if backend is not None:
   validation=json.loads((RUN/'qc/padding_validation.json').read_text())
   if backend!='coalesced_padding_v1' or runtime_sha!=filehash(ROOT/'inference_fast.py') or validation['optimized_source_sha256']!=runtime_sha or any(r['max_abs_error']>1e-5 for r in validation['checks']):raise RuntimeError('Unverified runtime backend')
  if actual.get('task_signature')!=task['signature'] or runtime!=expected:raise RuntimeError('Task provenance mismatch '+path.stem)
  d=pd.read_parquet(target);stage=task['stage']
  if 'cache_import' in actual:
   proof=actual['cache_import']
   for name in ['task','score','metadata']:
    if filehash(proof['source_'+name])!=proof['source_'+name+'_sha256']:raise RuntimeError('Imported source file changed')
   original_task=json.loads(Path(proof['source_task']).read_text());original_meta=json.loads(Path(proof['source_metadata']).read_text())
   if original_meta['task_signature']!=original_task['signature'] or original_meta['inference']!=actual['inference']:raise RuntimeError('Imported model/task definition changed')
   if stage=='coding':
    if task['records']!=original_task['records'] or filehash(target)!=proof['source_score_sha256'] or filehash(target.with_name(target.stem+'.raw.parquet'))!=proof['source_raw_sha256']:raise RuntimeError('Imported coding input/output changed')
   else:
    original=pd.read_parquet(proof['source_score']).set_index('variant_id').loc[d.variant_id].reset_index()
    if task['sequence']!=original_task['sequence'] or not d.equals(original):raise RuntimeError('Imported promoter input/output changed')
  if stage=='promoter':
   if d.variant_id.tolist()!=task['variant_ids'] or not (d.protein_id==task['protein_id']).all():raise RuntimeError('Promoter key mismatch '+path.stem)
   if not np.allclose(d.alt_pkd-d.ref_pkd,d.delta,atol=1e-7,rtol=0) or not np.allclose(d.delta.abs(),d.abs_delta,atol=1e-7,rtol=0):raise RuntimeError('Promoter difference mismatch')
  else:
   keys={(r['mapping_id'],s,n) for r in task['records'] for s in r['sources'] for n in [5,20,100]}
   if len(d)!=len(keys) or set(zip(d.mapping_id,d.source,d.n_dna))!=keys:raise RuntimeError('Coding key mismatch '+path.stem)
   raw=pd.read_parquet(target.with_name(target.stem+'.raw.parquet'))
   if len(raw)*3!=len(d):raise RuntimeError('Missing raw coding DNA predictions')
   lookup=d.set_index(['mapping_id','source','n_dna'])
   for r in raw.itertuples():
    ref=np.array(r.ref_pkd,dtype=np.float32);alt=np.array(r.alt_pkd,dtype=np.float32)
    if len(ref)!=100 or len(alt)!=100:raise RuntimeError('Truncated sampling predictions')
    for n in [5,20,100]:
     if not np.isclose(abs(alt[:n]-ref[:n]).mean(),lookup.loc[(r.mapping_id,r.source,n),'mean_abs_delta'],atol=1e-7,rtol=0):raise RuntimeError('Coding aggregation mismatch')
  numeric=d.select_dtypes(include='number')
  if not np.isfinite(numeric.to_numpy()).all():raise RuntimeError('Non-finite scores '+path.stem)
  counts[stage]+=1;rows[stage]+=len(d)
 save_json({'verified':True,'tasks':counts,'score_rows':rows,'prepared_inputs':inputs},RUN/'qc'/('full_output_verification.json' if stage_filter is None else stage_filter+'_output_verification.json'));return counts
if __name__=='__main__':print(verify())
