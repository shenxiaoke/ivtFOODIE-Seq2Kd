#!/usr/bin/env python3
"""Same FP32 model: coalesce identical padded protein query rows, then restore multiplicities."""
import types,fcntl,os
import torch
import inference
from common import *

def install_coalescing(engine):
 model=engine.model;old=model._apply_attention_block;blocks={id(b) for b in model.protein_cross_layers}
 if model.seq2contact:raise ValueError('Padding optimization requires seq2contact=False')
 def coalesced(self,x,block,residual=None,key=None,value=None,key_padding_mask=None):
  n=self._valid_protein_length
  # Retain the original path for extremely short truncations and full-width proteins.
  if id(block) in blocks and 20<=n<512:
   if residual is not None or x.shape[1]!=512:raise ValueError('Unexpected protein block')
   z,w=old(x[:,:n+1,:],block,residual=None,key=key,value=value,key_padding_mask=key_padding_mask)
   z=torch.cat([z[:,:n,:],z[:,n:n+1,:].expand(-1,512-n,-1)],dim=1)
   if w is not None:w=torch.cat([w[:,:n,:],w[:,n:n+1,:].expand(-1,512-n,-1)],dim=1)
   return z,w
  return old(x,block,residual=residual,key=key,value=value,key_padding_mask=key_padding_mask)
 model._apply_attention_block=types.MethodType(coalesced,model)
 return old

class Engine(inference.Engine):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);install_coalescing(self)
  self.signature=dict(self.signature,runtime_backend='coalesced_padding_v1',runtime_source_sha256=filehash(__file__))
 def embedding(self,seq):
  key=digest(seq)
  if key not in self.embcache:
   p=self.embedding_dir/(key+'.npy')
   with open(str(p)+'.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if p.exists() and not p.with_suffix('.json').exists():
     dest=RUN/'archive/embedding_orphans';dest.mkdir(parents=True,exist_ok=True)
     p.replace(dest/(key+'.'+str(os.getpid())+'.npy'))
  return super().embedding(seq)
 def predict(self,seq,dnas,cache=False):
  self.model._valid_protein_length=len(seq)
  return super().predict(seq,dnas,cache)

if __name__=='__main__':
 inference.Engine=Engine
 inference.main()
