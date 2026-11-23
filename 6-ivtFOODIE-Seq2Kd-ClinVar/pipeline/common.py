"""Shared, explicit conventions. No original data are modified."""
from pathlib import Path
import json, hashlib, os
import numpy as np
ROOT=Path(__file__).resolve().parent
RUN=Path(os.environ.get('CLINVAR_RUN_DIR',str(ROOT/'results'))).resolve()
WORKSPACE=Path(os.environ.get('CLINVAR_WORKSPACE',str(ROOT.parent/'inputs')))
SHARE=Path(os.environ.get('CLINVAR_SHARE', WORKSPACE))
ARCHIVE=Path(os.environ.get('CLINVAR_ARCHIVE', WORKSPACE/'data_archive/Kd_data_organization-eQTL_ClinVar/ClinVar'))
TFDIR=Path(os.environ.get('CLINVAR_TF_DIR', WORKSPACE/'human_tf'))
MODEL=ROOT/'assets/model'
PROBOUND=Path(os.environ.get('CLINVAR_PROBOUND', WORKSPACE/'probound'))
ESM=Path(os.environ.get('CLINVAR_ESM_CHECKPOINT', WORKSPACE/'esm2_t33_650M_UR50D.pt'))
LABELS=['Benign','Likely_benign','Likely_pathogenic','Pathogenic']
PCTS=[1,5,10,30]
AA='ACDEFGHIKLMNPQRSTVWY'
COMP=str.maketrans('ACGT','TGCA')
def rc(s):return s.translate(COMP)[::-1]
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save_json(obj,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str,allow_nan=False));t.replace(p)
def parquet(df,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial.parquet');df.to_parquet(t,index=False);t.replace(p)
def seeded_rng(key):return np.random.default_rng(int(digest('42:'+key)[:16],16))
def centered_crop(seq,index=None):
 if len(seq)<=512:return seq,0
 start=(len(seq)-512)//2 if index is None else max(0,min(index-256,len(seq)-512))
 return seq[start:start+512],start

def normalize_chromosome(chrom):
 return 'chrM' if chrom in ('MT','M','chrMT','chrM') else chrom if chrom.startswith('chr') else 'chr'+chrom
