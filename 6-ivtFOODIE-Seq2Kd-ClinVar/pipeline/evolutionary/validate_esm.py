#!/usr/bin/env python3
"""Independent scalar masked-forward checks against batched queue results."""
import torch,esm,numpy as np,pandas as pd
from shared import *
setup();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
path=ROOT/'assets/esm2_t33_650M_UR50D.pt';plan=json.loads((OUT/'qc/esm_plan.json').read_text())
assert filehash(path)==plan['signature']['esm_sha256']
data=torch.load(path,map_location='cpu',weights_only=False);m,a=esm.pretrained.load_model_and_alphabet_core(path.stem,data,regression_data=None);del data;m=m.cuda().eval();convert=a.get_batch_converter()
completed=[]
for p in (OUT/'scores').glob('*.json'):
 t=json.loads((OUT/'tasks'/p.name).read_text());completed.append((len(t['sequence']),t))
selected=[]
for size in [100,512,1022]:
 t=min(completed,key=lambda x:abs(x[0]-size))[1]
 if t['task_id'] not in [s['task_id'] for s in selected]:selected.append(t)
checks=[]
for t in selected:
 seq=t['sequence'];site=t['positions0'][0];_,_,tokens=convert([('wt',seq)]);tokens[0,site+1]=a.mask_idx
 with torch.inference_mode():logits=m(tokens.cuda(),return_contacts=False)['logits'][0,site+1].cpu().numpy()
 # Independent check: log softmax normalizers cancel, so raw-logit difference must agree.
 expected=np.array([logits[a.get_idx(x)]-logits[a.get_idx(seq[site])] for x in AA]);actual=pd.read_parquet(OUT/'scores'/(t['task_id']+'.parquet')).iloc[0][['llr_'+x for x in AA]].to_numpy(float)
 err=float(abs(expected-actual).max());assert err<2e-5,(len(seq),err)
 checks.append(dict(length=len(seq),site0=site,max_abs_error=err,method='scalar raw-logit differences vs batched log-softmax LLR'))
save_json(dict(passed=True,checkpoint_sha256=filehash(path),checks=checks),OUT/'qc/esm_scalar_validation.json');print(checks)
