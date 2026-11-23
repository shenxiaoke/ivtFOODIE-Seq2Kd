#!/usr/bin/env python3
"""FP32 v20_class_1, real sequence-specific ESM2 embeddings, paired invariant windows; durable queue."""
import argparse,os,sys,json,time,types,fcntl,socket,collections
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd,torch
from ruamel.yaml import YAML
from common import *

class Engine:
 def __init__(self,batch=128):
  torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
  sys.path.insert(0,str(MODEL))
  from deepproteindna.Bert import DeepProteinDNA_BERT
  self.acids='-XACDEFGHIKLMNPQRSTVWY';self.batch=batch
  cfg=YAML(typ='safe').load(MODEL/'deepproteindna_gen_test4.yaml')
  self.model=DeepProteinDNA_BERT(**cfg['model'],embeddings_path=None)
  state=torch.load(MODEL/'shenke_data/v20_class_1.pt',map_location='cpu',weights_only=False)
  state={k.removeprefix('module.'):v for k,v in state.items()};self.model.load_state_dict(state,strict=True)
  self.model.cuda().eval();self.esm=None;self.alphabet=None;self.converter=None;self.embcache=collections.OrderedDict();self.scorecache=collections.OrderedDict()
  self.embedding_dir=RUN/'embeddings';self.embedding_dir.mkdir(exist_ok=True)
  self.signature=json.loads((RUN/'qc/inference_signature.json').read_text())
 def embedding(self,seq):
  if not seq or len(seq)>512 or set(seq)-set(AA):raise ValueError('Invalid model protein sequence')
  key=digest(seq)
  if key in self.embcache:self.embcache.move_to_end(key);return self.embcache[key]
  p=self.embedding_dir/(key+'.npy')
  with open(str(p)+'.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not p.exists():
    if self.esm is None:
     import esm
     esmfile=ROOT/'assets/esm2_t33_650M_UR50D.pt'
     model_data=torch.load(esmfile,map_location='cpu',weights_only=False)
     self.esm,self.alphabet=esm.pretrained.load_model_and_alphabet_core(esmfile.stem,model_data,regression_data=None)
     del model_data
     self.esm=self.esm.cuda().eval();self.converter=self.alphabet.get_batch_converter()
    _,_,tok=self.converter([('sequence',seq)])
    with torch.inference_mode():rep=self.esm(tok.cuda(),repr_layers=[33],return_contacts=False)['representations'][33][:,1:len(seq)+1].cpu().numpy()
    tmp=p.with_suffix(f'.{os.getpid()}.tmp.npy');np.save(tmp,rep);tmp.replace(p)
    save_json({'sequence':seq,'sequence_sha256':key,'esm_sha256':self.signature['esm_sha256'],'layer':33,'bos_eos_removed':True},p.with_suffix('.json'))
   else:
    meta=json.loads(p.with_suffix('.json').read_text())
    if meta['sequence']!=seq or meta['esm_sha256']!=self.signature['esm_sha256']:raise ValueError('Embedding provenance mismatch')
   rep=np.load(p)
  if rep.shape!=(1,len(seq),1280) or not np.isfinite(rep).all():raise ValueError('Invalid embedding '+key)
  self.embcache[key]=rep
  while len(self.embcache)>16:self.embcache.popitem(last=False)
  return rep
 def predict(self,seq,dnas,cache=False):
  key=digest(seq+'|'+'|'.join(dnas))
  if cache and key in self.scorecache:self.scorecache.move_to_end(key);return self.scorecache[key].copy()
  rep=self.embedding(seq);pad=torch.zeros((1,512,1280),device='cuda');pad[:,:len(seq)]=torch.as_tensor(rep,device='cuda')
  with torch.inference_mode():
   fixed=self.model.transform_linear(pad) if self.model.transform_linear is not None else pad
   if self.model.segment_embedding:fixed=self.model.protein_segment_embedding(fixed,1)
  def fixed_encoder(model,protein_seq,protein_name=None):return fixed.expand(len(protein_seq),-1,-1)
  self.model._encode_protein=types.MethodType(fixed_encoder,self.model)
  aa=torch.tensor([[self.acids.index(x) for x in seq]+[0]*(512-len(seq))],device='cuda')
  lut={'A':2,'T':3,'C':4,'G':5};results=[]
  with torch.inference_mode():
   for start in range(0,len(dnas),self.batch):
    ss=dnas[start:start+self.batch]
    if any(len(x)!=40 or set(x)-set('ACGT') for x in ss):raise ValueError('Invalid DNA')
    dna=torch.tensor([[7]+[lut[b] for b in s] for s in ss],device='cuda')
    pred=self.model(dna,aa.expand(len(ss),-1),None,None,['sequence']*len(ss))[1]
    results.append(pred.cpu().numpy().ravel())
  value=np.concatenate(results)
  if len(value)!=len(dnas) or not np.isfinite(value).all():raise ValueError('Invalid predictions')
  if cache:
   self.scorecache[key]=value.copy()
   while len(self.scorecache)>64:self.scorecache.popitem(last=False)
  return value

def execute(engine,task,smoke=False):
 target=RUN/('smoke' if smoke else 'scores')/(task['task_id']+'.parquet');meta=target.with_suffix('.json')
 sig=dict(task_signature=task['signature'],inference=engine.signature)
 if target.exists() and meta.exists():
  if json.loads(meta.read_text())!=sig:raise ValueError('Output signature mismatch '+task['task_id'])
  return 'cached'
 if task['stage']=='coding':
  samples=pd.read_parquet(RUN/'prepared/sampled_dna.parquet');rows=[];raw=[]
  for r in task['records']:
   for source in r['sources']:
    d=samples[(samples.protein_id==r['protein_id'])&(samples.source==source)].sort_values('sample_index').dna_seq.tolist()
    assert len(d)==100
    ref=engine.predict(r['wt_sequence'],d,cache=True);alt=engine.predict(r['alt_sequence'],d)
    delta=alt-ref
    for n in [5,20,100]:rows.append(dict(mapping_id=r['mapping_id'],variant_id=r['variant_id'],protein_id=r['protein_id'],tf=r['tf'],source=source,n_dna=n,
      mean_abs_delta=float(np.abs(delta[:n]).mean()),abs_mean_delta=float(abs(delta[:n].mean())),mean_delta=float(delta[:n].mean()),max_abs_delta=float(np.abs(delta[:n]).max())))
    raw.append(dict(mapping_id=r['mapping_id'],source=source,ref_pkd=ref.tolist(),alt_pkd=alt.tolist()))
  parquet(pd.DataFrame(raw),target.with_name(target.stem+'.raw.parquet'));result=pd.DataFrame(rows)
 else:
  v=pd.read_parquet(RUN/'prepared/promoter_variants.parquet').set_index('variant_id').loc[task['variant_ids']]
  ref=engine.predict(task['sequence'],v.ref_seq.tolist());alt=engine.predict(task['sequence'],v.alt_seq.tolist())
  result=pd.DataFrame({'variant_id':v.index,'protein_id':task['protein_id'],'tf':task['tf'],'ref_pkd':ref,'alt_pkd':alt,'delta':alt-ref,'abs_delta':np.abs(alt-ref)})
 parquet(result,target);save_json(sig,meta);return len(result)

def main():
 p=argparse.ArgumentParser();p.add_argument('--task');p.add_argument('--smoke',action='store_true');p.add_argument('--batch',type=int,default=128);p.add_argument('--stage',choices=['coding','promoter','all'],default='all');args=p.parse_args()
 engine=Engine(args.batch);start=time.monotonic()
 if args.task:
  task=json.loads(Path(args.task).read_text());print(execute(engine,task,args.smoke),flush=True);return
 (RUN/'locks').mkdir(exist_ok=True);(RUN/'logs/workers').mkdir(parents=True,exist_ok=True)
 worker=f'{socket.gethostname()}.{os.getpid()}';fail=0
 while True:
  found=False
  tasks=sorted((RUN/'tasks').glob('*.json'))
  for path in tasks:
   taskid=path.stem
   if args.stage!='all' and not taskid.startswith(args.stage+'.'):continue
   target=RUN/'scores'/(taskid+'.json')
   if target.exists():continue
   with open(RUN/'locks'/(taskid+'.lock'),'w') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if target.exists():continue
    found=True;task=json.loads(path.read_text())
    save_json({'task':taskid,'worker':worker,'started':time.time()},RUN/'logs/workers'/(worker+'.json'))
    try:n=execute(engine,task)
    except Exception:
     import traceback
     save_json({'task':taskid,'worker':worker,'error':traceback.format_exc()},RUN/'logs/workers'/(worker+'.error.json'));raise
    print(taskid,n,f'elapsed={time.monotonic()-start:.1f}s',flush=True)
  if (RUN/'tasks/CLOSED').exists() and not found:
   relevant=[t for t in tasks if args.stage=='all' or t.stem.startswith(args.stage+'.')]
   if all((RUN/'scores'/(t.stem+'.json')).exists() for t in relevant):break
  if not found:time.sleep(10)
 print('Queue complete',flush=True)
if __name__=='__main__':main()
