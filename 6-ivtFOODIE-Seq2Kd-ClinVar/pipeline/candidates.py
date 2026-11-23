#!/usr/bin/env python3
"""Score unique promoter SNVs, select nested per-feature binding percentiles; CPU parallel."""
import argparse,json,collections,multiprocessing as mp,subprocess,tempfile
import numpy as np,pandas as pd
from numba import njit
from common import *
from motifs import encode

@njit(cache=True)
def score_matrix(encoded,mat,logsum=False):
 n=len(encoded);w=len(mat);out=np.empty(n)
 for i in range(n):
  best=-1e300
  for strand in range(2):
   for j in range(41-w):
    val=0.
    for k in range(w):
     b=encoded[i,j+k]
     val+=mat[k,b] if strand==0 else mat[w-1-k,3-b]
    if val>best:best=val
  if logsum:
   tot=0.
   for strand in range(2):
    for j in range(41-w):
     val=0.
     for k in range(w):
      b=encoded[i,j+k];val+=mat[k,b] if strand==0 else mat[w-1-k,3-b]
     tot+=np.exp(val-best)
   out[i]=(best+np.log(tot))/np.log(10.)
  else:out[i]=best
 return out

V=None;REF=None;ALT=None;SIGN=None

def run_one(job):
 source,feature,tf,protein,mat,logsum=job
 path=RUN/f'candidates/{source}/{feature}.parquet';meta=path.with_suffix('.json')
 signature=dict(SIGN,job_hash=digest(json.dumps([source,feature,tf,protein,mat.tolist(),logsum])))
 if path.exists() and meta.exists() and json.loads(meta.read_text())==signature:return source,feature,'cached'
 rs=score_matrix(REF,mat,logsum);alts=score_matrix(ALT,mat,logsum);binding=np.maximum(rs,alts)
 cuts={pct:float(np.percentile(binding,100-pct)) for pct in PCTS};keep=binding>=cuts[30]
 d=V.loc[keep,['variant_id']].copy();d['ref_score']=rs[keep];d['alt_score']=alts[keep];d['tf']=tf;d['protein_id']=protein;d['feature_id']=feature
 for pct in PCTS:d[f'top{pct}']=binding[keep]>=cuts[pct]
 d['delta']=d.alt_score-d.ref_score;d['abs_delta']=d.delta.abs()
 parquet(d,path);save_json(signature,meta);save_json(dict(thresholds=cuts,universe=len(V),counts={f'top{pct}':int(d[f'top{pct}'].sum()) for pct in PCTS}),path.with_suffix('.thresholds.json'))
 return source,feature,len(d)

def probound_jobs(proteins):
 data=json.loads((PROBOUND/'MotifCentral.v1.0.0.json').read_text());allowed=set(proteins.tf.str.upper());jobs=[];manifest=[]
 for m in data:
  factors=sorted(set(f['gene_symbol'].upper() for f in m['metadata']['factors'] if str(f.get('tax_id'))=='9606'))
  genes=set(factors)&allowed
  if not genes:continue
  modes=m['modelSettings']['bindingModes']
  if len(modes)!=1 or modes[0]['dinucleotideDistance']!=0 or modes[0]['flankLength']!=0 or modes[0].get('singleStrand',False):raise ValueError('Unsupported ProBound architecture')
  if len(factors)!=1:continue # A multi-factor fit cannot be attributed to an individual TF.
  tf=factors[0];fit=str(m['metadata']['fit_id']);mat=np.array(m['coefficients']['bindingModes'][0]['mononucleotide']).reshape(-1,4)
  if len(mat)>40:continue
  jobs.append(('probound',tf+'.'+fit,tf,'',mat,True));manifest.append(dict(tf=tf,fit_id=fit,metadata=m['metadata'],mode=modes[0]))
 save_json(manifest,RUN/'qc/probound_model_selection.json');return jobs

def validate_probound(jobs):
 # Verify optimized sum-over-offsets/both-strands against the supplied Java engine.
 folder=RUN/'qc/probound_java_validation';folder.mkdir(parents=True,exist_ok=True)
 seqs=V.ref_seq.iloc[:32].tolist()+V.alt_seq.iloc[:32].tolist();inp=folder/'input.txt';inp.write_text('\n'.join(seqs)+'\n');encoded=encode(seqs)
 jar=PROBOUND/'ProBoundTools_new/ProBoundTools/ProBoundTools/target/ProBound-jar-with-dependencies.jar'
 java=PROBOUND/'jdk-23.0.1/bin/java';checks=[]
 chosen=[jobs[i] for i in sorted(set([0,len(jobs)//2,len(jobs)-1]))]
 for job in chosen:
  _,feature,tf,pid,mat,ls=job;fit=feature.rsplit('.',1)[1];out=folder/(feature+'.txt')
  cmd=f'loadModelFromPortalJSON({PROBOUND}/MotifCentral.v1.0.0.json, {fit}).addNScoring().inputTXT({inp}).bindingModeScores({out})'
  result=subprocess.run([str(java),'-cp',str(jar),'proBoundTools/App','-c',cmd],capture_output=True,text=True,check=True)
  (folder/(feature+'.log')).write_text(result.stdout+result.stderr)
  d=pd.read_csv(out,sep='\t',header=None)
  if d.shape!=(len(seqs),2) or d[0].tolist()!=seqs:raise ValueError('Java alignment mismatch '+feature)
  js=np.log10(d[1].to_numpy());ours=score_matrix(encoded,mat,True);err=float(np.max(np.abs(js-ours)))
  if not np.allclose(js,ours,atol=5e-6,rtol=0):raise ValueError(f'ProBound Java mismatch {feature}: {err}')
  checks.append(dict(feature=feature,n=len(seqs),max_abs_log10_error=err))
 save_json(checks,folder/'validation.json')

def main():
 global V,REF,ALT,SIGN
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);p.add_argument('--include-probound',action='store_true');args=p.parse_args()
 V=pd.read_parquet(RUN/'prepared/promoter_variants.parquet');REF=encode(V.ref_seq);ALT=encode(V.alt_seq)
 proteins=pd.read_parquet(RUN/'prepared/proteins.parquet');motifs=json.loads((RUN/'prepared/motifs.json').read_text())
 SIGN=dict(variants=filehash(RUN/'prepared/promoter_variants.parquet'),script=filehash(__file__),pcts=PCTS,universe='unique promoter SNVs, no clinical labels used')
 jobs=[]
 for m in motifs:
  mat=np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
  jobs.append((m['source'],m['protein_id'],m['tf'],m['protein_id'],mat,False))
 if args.include_probound:
  prob=probound_jobs(proteins);validate_probound(prob);jobs+=prob
 save_json({'total':len(jobs),'counts':dict(collections.Counter(j[0] for j in jobs))},RUN/'qc/candidate_expected.json')
 # JIT before fork, avoids 16 redundant compiler invocations.
 score_matrix(REF[:1],jobs[0][4],False)
 with mp.get_context('fork').Pool(args.workers) as pool:
  for i,result in enumerate(pool.imap_unordered(run_one,jobs)):
   if i%25==0:print(i+1,len(jobs),result,flush=True)
 save_json({'complete':True,'expected':len(jobs)},RUN/'qc/candidates_complete.json')
if __name__=='__main__':main()
