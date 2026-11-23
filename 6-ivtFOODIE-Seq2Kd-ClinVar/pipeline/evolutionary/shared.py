"""Isolated evolutionary analyses; original Seq2Kd inputs and code stay immutable."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import *
OUT=RUN/'evolutionary'
DB=Path(os.environ.get('CLINVAR_AM_DIR', WORKSPACE/'AlphaMissense_database'))
def setup():
 for d in ['tables','figures','tasks','scores','locks','logs','qc']: (OUT/d).mkdir(parents=True,exist_ok=True)
def predicted_coding():
 import pandas as pd
 c=pd.read_parquet(RUN/'prepared/coding_variants.parquet')
 ids=set()
 for p in (RUN/'tasks').glob('coding.*.json'):
  ids.update(r['mapping_id'] for r in json.loads(p.read_text())['records'])
 return c[c.mapping_id.isin(ids)].copy()
