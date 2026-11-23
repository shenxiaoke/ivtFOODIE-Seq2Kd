#!/usr/bin/env python3
"""Real-sequence end-to-end control: WT=WT, changed mutant embedding, original encoder equivalence."""
import types,time,os,socket
import pandas as pd,numpy as np,torch
from common import *
from inference import Engine
start=time.monotonic();engine=Engine()
v=pd.read_parquet(RUN/'prepared/promoter_variants.parquet');dna=v.ref_seq.iloc[:32].tolist()
c=pd.read_parquet(RUN/'prepared/coding_variants.parquet');r=c[(c.status=='ready')&(c.consequence=='missense')&(c.wt_sequence.str.len().between(80,200))].iloc[0]
wt=engine.predict(r.wt_sequence,dna);same=engine.predict(r.wt_sequence,dna);alt=engine.predict(r.alt_sequence,dna)
if not np.array_equal(wt,same):raise ValueError('WT control not reproducible')
a=engine.embedding(r.wt_sequence);b=engine.embedding(r.alt_sequence)
if np.array_equal(a,b) or np.array_equal(wt,alt):raise ValueError('Mutation has no embedding/prediction change')
# Restore original encoder and supply the newly generated exact WT embedding.
from deepproteindna.Bert import DeepProteinDNA_BERT
engine.model._encode_protein=types.MethodType(DeepProteinDNA_BERT._encode_protein,engine.model)
engine.model.embeddings_dict={'validation':torch.as_tensor(a,device='cuda')}
aa=torch.tensor([[engine.acids.index(x) for x in r.wt_sequence]+[0]*(512-len(r.wt_sequence))],device='cuda')
lut={'A':2,'T':3,'C':4,'G':5};tokens=torch.tensor([[7]+[lut[b] for b in s] for s in dna],device='cuda')
with torch.inference_mode():original=engine.model(tokens,aa.expand(len(dna),-1),None,None,['validation']*len(dna))[1].cpu().numpy().ravel()
err=float(abs(original-wt).max())
if not np.allclose(original,wt,atol=1e-5,rtol=0):raise ValueError('Original encoder mismatch')
output=RUN/'qc'/os.environ.get('CLINVAR_VALIDATION_NAME','inference_validation.json')
save_json(dict(hostname=socket.gethostname(),n_dna=len(dna),mapping_id=r.mapping_id,variant_id=r.variant_id,protein_id=r.protein_id,wt_identical_control=True,
 embedding_max_abs_difference=float(abs(a-b).max()),mutation_max_abs_delta_pkd=float(abs(alt-wt).max()),original_encoder_max_abs_difference=err,elapsed_seconds=time.monotonic()-start),output)
print((output).read_text(),flush=True)
