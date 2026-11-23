"""Audit removal of masked protein keys/values; no parameters, precision or valid residues changed."""
import types,time
import numpy as np,pandas as pd,torch
from common import *
from inference import Engine
from inference_fast import install_coalescing
engine=Engine();v=pd.read_parquet(RUN/'prepared/promoter_variants.parquet');dnas=v.ref_seq.iloc[:32].tolist()+v.alt_seq.iloc[:32].tolist()
p=pd.read_parquet(RUN/'prepared/proteins.parquet');seq=p[p.promoter_sequence.str.len()==512].iloc[0].promoter_sequence
baseline=engine.model._apply_attention_block;install_coalescing(engine);v1=engine.model._apply_attention_block;dna_blocks={id(b) for b in engine.model.dna_cross_layers}
def trim(model,x,block,residual=None,key=None,value=None,key_padding_mask=None):
 n=model._valid_protein_length
 if id(block) in dna_blocks and 20<=n<512:
  kv=key[:,:n,:];vv=kv if key is value else value[:,:n,:]
  z,w=v1(x,block,residual=residual,key=kv,value=vv,key_padding_mask=key_padding_mask[:,:n])
  if w is not None:w=torch.nn.functional.pad(w,(0,512-n))
  return z,w
 return v1(x,block,residual=residual,key=key,value=value,key_padding_mask=key_padding_mask)
v2=types.MethodType(trim,engine.model);rows=[]
for n in [1,20,80,200,300,511,512]:
 s=seq[:n];engine.model._valid_protein_length=n;engine.embedding(s);vals=[];times=[]
 for implementation in [baseline,v1,v2]:
  engine.model._apply_attention_block=implementation;torch.cuda.synchronize();start=time.monotonic();vals.append(engine.predict(s,dnas));torch.cuda.synchronize();times.append(time.monotonic()-start)
 error=float(abs(vals[2]-vals[0]).max())
 if error>1e-5:raise RuntimeError(f'Numerical mismatch: {n}: {error}')
 rows.append(dict(length=n,n_dna=len(dnas),max_abs_error_vs_original=error,seconds_original=times[0],seconds_v1=times[1],seconds_v2=times[2]));print(rows[-1],flush=True)
save_json(rows,RUN/'qc/masked_key_benchmark.json')
