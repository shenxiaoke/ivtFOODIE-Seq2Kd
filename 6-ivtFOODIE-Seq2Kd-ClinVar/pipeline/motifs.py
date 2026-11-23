"""MEME probability matrices, independent sampling, double-strand log-odds scanning."""
import re
import numpy as np
from common import *
def parse_meme(path):
 lines=Path(path).read_text().splitlines();bg=np.full(4,.25);result=[];name=None
 for i,s in enumerate(lines):
  if s.startswith('Background letter frequencies'):
   m=re.findall(r'([ACGT])\s+([0-9.eE+-]+)',lines[i+1]);d=dict(m)
   if len(d)==4:bg=np.array([float(d[x]) for x in 'ACGT'])
  if s.startswith('MOTIF '):name=s.split()[1]
  if s.startswith('letter-probability matrix:'):
   w=int(re.search(r'\bw=\s*(\d+)',s)[1]);p=np.array([[float(x) for x in lines[j].split()[:4]] for j in range(i+1,i+1+w)])
   if p.shape!=(w,4) or (p<0).any() or not np.allclose(p.sum(1),1,atol=.002):raise ValueError(path)
   p=p/p.sum(1,keepdims=True)
   result.append({'motif_id':name,'probabilities':p.tolist(),'background':(bg/bg.sum()).tolist(),'path':str(path)})
 return result
def load_motifs(proteins):
 streme={};duplicates=[]
 base=ARCHIVE/'dna_mutation_sk/Seq2Kd_motif'
 for d in sorted(base.glob('streme_out_100.part[12]')):
  for p in sorted(d.glob('*/streme.txt')):
   pid=p.parent.name.split('_')[0];m=parse_meme(p)
   if m:
    if pid in streme:
     duplicates.append(dict(protein_id=pid,selected=streme[pid]['path'],excluded=str(p),rule='part1 before part2; first archive path wins'));continue
    streme[pid]=m[0]
 save_json(duplicates,RUN/'qc/duplicate_streme_motifs.json')
 out=[]
 for r in proteins.itertuples():
  motif=streme.get(r.protein_id)
  if motif and len(motif['probabilities'])<=40:out.append(dict(motif,source='streme',protein_id=r.protein_id,tf=r.tf))
 return out
def sample_dna(m,n=100):
 rng=seeded_rng('sample:'+m['source']+':'+m['protein_id']);p=np.array(m['probabilities']);bg=np.array(m['background']);w=len(p)
 dna=rng.choice(np.array(list('ACGT')),(n,40),p=bg)
 left=(40-w)//2
 for j,row in enumerate(p):dna[:,left+j]=rng.choice(np.array(list('ACGT')),n,p=row)
 return [''.join(x) for x in dna]
def encode(seqs):
 lut=np.full(256,255,np.uint8)
 for i,b in enumerate(b'ACGT'):lut[b]=i
 a=lut[np.array(seqs,dtype='S40').view(np.uint8).reshape(-1,40)]
 if (a>3).any():raise ValueError('Non ACGT or non-40bp DNA')
 return a
def pwm_scores(encoded,m):
 p=np.array(m['probabilities']);bg=np.array(m['background']);pwm=np.log2((p+1e-6)/bg)
 result=np.full(len(encoded),-np.inf)
 for mat in (pwm,pwm[::-1,::-1]):
  for offset in range(41-len(mat)):
   s=np.zeros(len(encoded))
   for j in range(len(mat)):s+=mat[j,encoded[:,offset+j]]
   result=np.maximum(result,s)
 return result
