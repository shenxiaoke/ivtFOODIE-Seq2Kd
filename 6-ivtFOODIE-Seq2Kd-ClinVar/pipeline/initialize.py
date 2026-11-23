#!/usr/bin/env python3
"""Stage shared-node model assets and establish an immutable inference/environment signature."""
import sys,shutil,importlib.metadata as md
from common import *

def main():
 (RUN/'qc').mkdir(parents=True,exist_ok=True);(ROOT/'assets').mkdir(exist_ok=True)
 old=Path(os.environ.get('CLINVAR_MODEL_SOURCE', WORKSPACE/'Seq2Kd-model'))
 if not (MODEL/'deepproteindna/Bert.py').exists():shutil.copytree(old/'deepproteindna',MODEL/'deepproteindna',dirs_exist_ok=True)
 for name in ['deepproteindna_gen_test4.yaml','shenke_data/v20_class_1.pt']:
  dest=MODEL/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(old/name,dest)
 dest=ROOT/'assets/esm2_t33_650M_UR50D.pt'
 if not dest.exists():shutil.copy2(ESM,dest)
 signature=dict(checkpoint_sha256=filehash(MODEL/'shenke_data/v20_class_1.pt'),esm_sha256=filehash(dest),model_config_sha256=filehash(MODEL/'deepproteindna_gen_test4.yaml'),model_source_sha256=filehash(MODEL/'deepproteindna/Bert.py'),inference_source_sha256=filehash(ROOT/'inference.py'),sampling_seed=42,precision='FP32',batch_size=128,esm_layer=33,crop_policy='coding mutation-centered shared window; promoter fixed centered window')
 path=RUN/'qc/inference_signature.json'
 if path.exists() and json.loads(path.read_text())!=signature:raise RuntimeError('Inference definition changed; choose a NEW CLINVAR_RUN_DIR instead of mixing results')
 save_json(signature,path)
 packages=['numpy','pandas','scipy','scikit-learn','statsmodels','pysam','biopython','numba','pyarrow','torch','fair-esm','ruamel.yaml','matplotlib','transformers','torch-geometric']
 save_json({'python':sys.version,'executable':sys.executable,'packages':{p:md.version(p) for p in packages}},RUN/'qc/environment.json')
 (RUN/'qc/requirements.lock.txt').write_text('\n'.join(p+'=='+md.version(p) for p in packages)+'\n')
 files=[p for p in MODEL.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.pt')]
 save_json([{'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':filehash(p)} for p in files],RUN/'qc/model_bundle_manifest.json')
 print('Initialized',RUN,flush=True)
if __name__=='__main__':main()
