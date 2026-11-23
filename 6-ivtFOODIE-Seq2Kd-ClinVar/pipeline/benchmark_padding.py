import sys,time,types
import torch,numpy as np,pandas as pd
from common import *
from inference import Engine
engine=Engine();v=pd.read_parquet(RUN/'prepared/promoter_variants.parquet');dnas=v.ref_seq.iloc[:128].tolist();p=pd.read_parquet(RUN/'prepared/proteins.parquet');p=p[p.promoter_sequence.str.len().between(80,200)].iloc[0];seq=p.promoter_sequence
# Populate ESM cache before timing either implementation.
engine.embedding(seq)
old=engine.model._apply_attention_block;blocks={id(b) for b in engine.model.protein_cross_layers}
def compressed(model,x,block,residual=None,key=None,value=None,key_padding_mask=None):
 n=len(seq)
 if id(block) in blocks and n<512:
  z,w=old(x[:,:n+1,:],block,residual=None,key=key,value=value,key_padding_mask=key_padding_mask)
  z=torch.cat([z[:,:n,:],z[:,n:n+1,:].expand(-1,512-n,-1)],dim=1)
  if w is not None:w=torch.cat([w[:,:n,:],w[:,n:n+1,:].expand(-1,512-n,-1)],dim=1)
  return z,w
 return old(x,block,residual=residual,key=key,value=value,key_padding_mask=key_padding_mask)
base=[];fast=[];errs=[]
for i in range(3):
 engine.model._apply_attention_block=old;torch.cuda.synchronize();start=time.monotonic();a=engine.predict(seq,dnas);torch.cuda.synchronize();base.append(time.monotonic()-start)
 engine.model._apply_attention_block=types.MethodType(compressed,engine.model);torch.cuda.synchronize();start=time.monotonic();b=engine.predict(seq,dnas);torch.cuda.synchronize();fast.append(time.monotonic()-start);errs.append(float(abs(a-b).max()))
print(dict(protein=p.protein_id,length=len(seq),base_seconds=base,coalesced_seconds=fast,max_abs_errors=errs),flush=True)
save_json(dict(protein=p.protein_id,length=len(seq),base_seconds=base,coalesced_seconds=fast,max_abs_errors=errs),RUN/'qc/padding_benchmark.json')
