"""Numerical/performance audit of coalescing identical padded protein query rows (no model change)."""
import types,time
import numpy as np,pandas as pd,torch
from common import *
from inference import Engine
engine=Engine();v=pd.read_parquet(RUN/'prepared/promoter_variants.parquet');dnas=v.ref_seq.iloc[:16].tolist()+v.alt_seq.iloc[:16].tolist()
p=pd.read_parquet(RUN/'prepared/proteins.parquet');seq=p[p.promoter_sequence.str.len()==512].iloc[0].promoter_sequence
from inference_fast import install_coalescing
old=install_coalescing(engine);fast_method=engine.model._apply_attention_block
checks=[]
for length in [1,2,20,80,160,200,300,511,512]:
 s=seq[:length];engine.model._valid_protein_length=length;engine.embedding(s);engine.model._apply_attention_block=old;torch.cuda.synchronize();t=time.monotonic();a=engine.predict(s,dnas);torch.cuda.synchronize();baseline=time.monotonic()-t
 engine.model._apply_attention_block=fast_method;torch.cuda.synchronize();t=time.monotonic();b=engine.predict(s,dnas);torch.cuda.synchronize();fast=time.monotonic()-t;err=float(abs(a-b).max())
 if not np.allclose(a,b,atol=1e-5,rtol=0):raise RuntimeError('Numerical mismatch')
 checks.append(dict(length=length,n_dna=len(dnas),max_abs_error=err,baseline_seconds=baseline,coalesced_seconds=fast));print(checks[-1],flush=True)
save_json({'purpose':'numeric stress tests on real fragment prefixes, not biological variant predictions','checks':checks,'atol':1e-5,'rtol':0,'inference_source_sha256':filehash(ROOT/'inference.py'),'optimized_source_sha256':filehash(ROOT/'inference_fast.py')},RUN/'qc/padding_validation.json')

save_json({'baseline':json.loads((RUN/'qc/inference_signature.json').read_text()),'coalesced_padding_v1':{'source_sha256':filehash(ROOT/'inference_fast.py'),'validation_sha256':filehash(RUN/'qc/padding_validation.json'),'max_abs_validation_error':max(x['max_abs_error'] for x in checks),'precision':'FP32','original_path_for_lengths':'<20 or 512'}},RUN/'qc/runtime_backends.json')
