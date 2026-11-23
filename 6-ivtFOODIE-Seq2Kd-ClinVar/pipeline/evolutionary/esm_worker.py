#!/usr/bin/env python3
"""One masked residue per sequence; evaluate 20 canonical AA log probabilities, FP32."""
import os,time,fcntl,socket,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,pandas as pd,torch,esm
from shared import *
def main():
 setup();ap=argparse.ArgumentParser();ap.add_argument('--max-tasks',type=int);ap.add_argument('--batch',type=int,default=8);args=ap.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 checkpoint=ROOT/'assets/esm2_t33_650M_UR50D.pt';data=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model,alphabet=esm.pretrained.load_model_and_alphabet_core(checkpoint.stem,data,regression_data=None);del data
 model=model.cuda().eval();convert=alphabet.get_batch_converter();aaindices=torch.tensor([alphabet.get_idx(a) for a in AA],device='cuda');done=0;start=time.time()
 for path in sorted((OUT/'tasks').glob('*.json')):
  target=OUT/'scores'/path.name
  if target.exists():continue
  with open(OUT/'locks'/(path.stem+'.lock'),'a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:continue
   if target.exists():continue
   task=json.loads(path.read_text())
   if task['signature']['code_sha256']!=filehash(__file__):raise RuntimeError('Worker changed after task creation')
   seq=task['sequence'];positions=task['positions0'];rows=[];batch=min(args.batch,max(1,4096//(len(seq)+2)))
   for offset in range(0,len(positions),batch):
    sites=positions[offset:offset+batch];_,_,tokens=convert([('wt',seq)]*len(sites));tokens=tokens.cuda();ix=torch.arange(len(sites),device='cuda');pp=torch.tensor(sites,device='cuda')+1;tokens[ix,pp]=alphabet.mask_idx
    with torch.inference_mode():
     logits=model(tokens,return_contacts=False)['logits'][ix,pp];logp=logits.log_softmax(-1).index_select(1,aaindices);refids=torch.tensor([AA.index(seq[i]) for i in sites],device='cuda');llr=(logp-logp[ix,refids,None]).cpu().numpy()
    if not np.isfinite(llr).all():raise ValueError('Nonfinite masked LLR')
    for i,values in zip(sites,llr):rows.append(dict(sequence_key=task['sequence_key'],site0=i,ref_aa=seq[i],**{'llr_'+a:float(v) for a,v in zip(AA,values)}))
   parquet(pd.DataFrame(rows),target.with_suffix('.parquet'));save_json(dict(task_hash=task['task_hash'],signature=task['signature'],worker=f'{socket.gethostname()}.{os.getpid()}',sites=len(rows)),target);done+=1
   print('DONE',path.stem,'masked',len(rows),'length',len(seq),'elapsed',round(time.time()-start,1),flush=True)
   if args.max_tasks and done>=args.max_tasks:break
 print('Worker queue pass complete',done,flush=True)
if __name__=='__main__':main()
