#!/usr/bin/env python3
"""Verify masked outputs, aggregate truncation proxy, retain each variant's DNA profile metrics."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
from shared import *

def seq_metrics(ref,alt):
 d=alt-ref
 i,j=np.triu_indices(len(ref),1);refgap=ref[i]-ref[j];altgap=alt[i]-alt[j];eligible=(abs(refgap)>4e-5)&(abs(altgap)>4e-5);reversed_pairs=eligible&(refgap*altgap<0);npairs=int(eligible.sum())
 rho=float(spearmanr(ref,alt).statistic) if np.ptp(ref)>0 and np.ptp(alt)>0 else np.nan
 return dict(robust_reversal_fraction=float(reversed_pairs.sum()/npairs) if npairs else np.nan,robust_eligible_DNA_pairs=npairs,mean_abs_delta=np.abs(d).mean(),mean_delta=d.mean(),sd_delta=d.std(ddof=1),range90_delta=np.quantile(d,.95)-np.quantile(d,.05),dna_rank_loss=1-rho,abs_mean_delta=abs(d.mean()),fraction_gain=(d>1e-5).mean(),fraction_loss=(d< -1e-5).mean())

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--dna-only',action='store_true');args=ap.parse_args()
 setup()
 validation=json.loads((OUT/'qc/esm_scalar_validation.json').read_text())
 if not validation['passed'] or validation['checkpoint_sha256']!=filehash(ROOT/'assets/esm2_t33_650M_UR50D.pt'):raise RuntimeError('Masked forward validation/checkpoint mismatch')
 c=predicted_coding();lookup=c.set_index('mapping_id');rows=[];profiles={};example=[]
 for p in sorted((RUN/'scores').glob('coding.*.raw.parquet')):
  for r in pd.read_parquet(p).itertuples():
   ref=np.asarray(r.ref_pkd);alt=np.asarray(r.alt_pkd)
   if ref.shape!=(100,) or alt.shape!=(100,):raise ValueError('Incomplete DNA panel')
   x=lookup.loc[r.mapping_id];rows.append(dict(mapping_id=r.mapping_id,source=r.source,**seq_metrics(ref,alt)))
   profiles.setdefault((r.source,x.protein_id,x.consequence),[]).append((r.mapping_id,alt-ref))
 metrics=pd.DataFrame(rows);interaction={};variance=[]
 for (source,pid,consequence),group in profiles.items():
  matrix=np.array([x[1] for x in group]);grand=matrix.mean();vm=matrix.mean(1,keepdims=True);dm=matrix.mean(0,keepdims=True);resid=matrix-vm-dm+grand
  for (mid,d),v in zip(group,np.sqrt((resid**2).mean(1))):interaction[(mid,source)]=v
  tot=((matrix-grand)**2).sum();vss=100*((vm-grand)**2).sum();dss=len(matrix)*((dm-grand)**2).sum();iss=(resid**2).sum()
  variance.append(dict(source=source,protein_id=pid,consequence=consequence,n_variants=len(matrix),variant_shift_ss=vss,dna_shared_ss=dss,interaction_ss=iss,total_ss=tot,interaction_fraction=iss/tot if tot else np.nan))
 metrics['interaction_rms']=[interaction[(m,s)] for m,s in zip(metrics.mapping_id,metrics.source)]
 merged=metrics.merge(c.drop(columns=['wt_sequence','alt_sequence']),on='mapping_id',validate='many_to_one')
 parquet(merged,OUT/'tables/seq2kd_dna_metrics.parquet');merged.to_csv(OUT/'tables/seq2kd_dna_metrics.csv',index=False)
 pd.DataFrame(variance).to_csv(OUT/'tables/dna_variance_decomposition.csv',index=False)
 # Example selected by largest missense cohort, without clinical label or effect selection.
 examples={}
 for source in ['streme']:
  eligible=[(k,g) for k,g in profiles.items() if k[0]==source and k[2]=='missense' and len(g)>=10]
  if eligible:
   k,g=max(eligible,key=lambda x:(len(x[1]),x[0][1]));matrix=np.array([x[1] for x in g]);np.savez_compressed(OUT/'tables'/f'dna_profile_example.{source}.npz',delta=matrix,mapping_id=np.array([x[0] for x in g]),protein_id=k[1]);examples[source]=dict(protein_id=k[1],n_variants=len(g))
 save_json(examples,OUT/'qc/profile_examples.json')
 if args.dna_only:
  print('DNA metrics complete',len(metrics),flush=True);return
 # Fail closed on missing task, wrong source or changed input/checkpoint.
 scoreframes=[];taskfiles=sorted((OUT/'tasks').glob('*.json'));plan=json.loads((OUT/'qc/esm_plan.json').read_text())
 if len(taskfiles)!=plan['tasks']:raise RuntimeError('Task count changed')
 for p in taskfiles:
  task=json.loads(p.read_text());meta=OUT/'scores'/p.name;data=meta.with_suffix('.parquet')
  if not meta.exists() or not data.exists():raise RuntimeError('Incomplete ESM task '+p.stem)
  md=json.loads(meta.read_text())
  if md['task_hash']!=task['task_hash'] or md['signature']!=plan['signature']:raise RuntimeError('ESM provenance mismatch')
  frame=pd.read_parquet(data)
  if frame.site0.tolist()!=task['positions0'] or set(frame.sequence_key)!={task['sequence_key']}:raise RuntimeError('Masked output positions mismatch')
  v=frame[['llr_'+a for a in AA]].to_numpy()
  if not np.isfinite(v).all() or not np.allclose([frame.iloc[i]['llr_'+r] for i,r in enumerate(frame.ref_aa)],0,atol=1e-7):raise ValueError('Invalid LLR normalization')
  scoreframes.append(frame)
 scores=pd.concat(scoreframes,ignore_index=True)
 if scores.duplicated(['sequence_key','site0']).any():raise ValueError('Duplicate masked sites')
 parquet(scores,OUT/'tables/esm_all_masked_sites.parquet');byseq={key:g.set_index('site0') for key,g in scores.groupby('sequence_key')};links=pd.read_parquet(OUT/'tables/esm_links.parquet');out=[]
 for r in links.itertuples():
  x=lookup.loc[r.mapping_id];g=byseq[r.sequence_key]
  if x.consequence=='missense':
   llr=float(g.loc[r.site0,'llr_'+x.alt_aa]);out.append(dict(mapping_id=r.mapping_id,context=r.context,esm_llr=llr,evolutionary_score=-llr,score_kind='negative_missense_masked_LLR',deleted_residues=0,deleted_fraction=0.,worst_site0=np.nan,worst_alt_aa='',q05_deleterious_proxy=np.nan))
  else:
   z=g.loc[range(r.site0,r.deleted_end0)].copy();values=z[['llr_'+a for a in AA]].to_numpy();refs=[AA.index(a) for a in z.ref_aa]
   values[np.arange(len(z)),refs]=np.inf;flat=np.argmin(values);ii,jj=np.unravel_index(flat,values.shape);mn=float(values[ii,jj]);finite=values[np.isfinite(values)]
   out.append(dict(mapping_id=r.mapping_id,context=r.context,esm_llr=np.nan,evolutionary_score=-mn,score_kind='worst_single_substitution_in_deleted_DBD_proxy',deleted_residues=len(z),deleted_fraction=len(z)/len(lookup.loc[r.mapping_id,'wt_sequence']),worst_site0=int(z.index[ii]),worst_alt_aa=AA[jj],q05_deleterious_proxy=float(-np.quantile(finite,.05))))
 esmout=pd.DataFrame(out).merge(links,on=['mapping_id','context'],validate='one_to_one')
 esmout['context_window_start0']=[(lookup.loc[mid,'fragment_start0']+lookup.loc[mid,'crop_start0']) if context!='full_or_1022_window' else start for mid,context,start in zip(esmout.mapping_id,esmout.context,esmout.full_window_start0)]
 esmout['worst_aa_pos1']=esmout.context_window_start0+esmout.worst_site0+1
 esmout['worst_ref_aa']=[lookup.loc[mid,'wt_sequence'][int(i)] if np.isfinite(i) else '' for mid,i in zip(esmout.mapping_id,esmout.worst_site0)]
 resolution=[]
 for tf,g in esmout[esmout.context=='dbd_deleted_region'].merge(c[['mapping_id','tf','variant_id']],on='mapping_id').groupby('tf'):
  bysnp=g.groupby('variant_id').evolutionary_score.mean();resolution.append(dict(tf=tf,n_snps=len(bysnp),distinct_scores=bysnp.nunique(),largest_tie_count=int(bysnp.value_counts().max()),score_min=bysnp.min(),score_max=bysnp.max()))
 pd.DataFrame(resolution).to_csv(OUT/'tables/stop_gain_proxy_resolution.csv',index=False)
 parquet(esmout,OUT/'tables/esm_variant_scores.parquet');esmout.to_csv(OUT/'tables/esm_variant_scores.csv',index=False)
 am=pd.read_parquet(OUT/'tables/alphamissense_mapping.parquet')[['mapping_id','am_score_validated','am_mapping_status']]
 final=merged.merge(esmout,on='mapping_id',validate='many_to_many').merge(am,on='mapping_id',how='left',validate='many_to_one')
 final.loc[final.consequence!='missense','am_mapping_status']='not_applicable_non_missense'
 parquet(final,OUT/'tables/association_input.parquet');final.to_csv(OUT/'tables/association_input.csv',index=False)
 if not np.allclose(pd.DataFrame(variance).eval('variant_shift_ss+dna_shared_ss+interaction_ss'),pd.DataFrame(variance).total_ss,atol=1e-7,rtol=1e-10):raise ValueError('DNA variance decomposition does not reconcile')
 save_json(dict(verified=True,masked_sites=len(scores),tasks=len(taskfiles),variant_scores=len(esmout),predicted_unique_snps=c.variant_id.nunique(),source_hashes={str(p):filehash(p) for p in [HERE/'assemble.py',HERE/'esm_worker.py',OUT/'tables/esm_links.parquet']}),OUT/'qc/assembly_complete.json');print('Assembled',len(final),flush=True)
if __name__=='__main__':main()
