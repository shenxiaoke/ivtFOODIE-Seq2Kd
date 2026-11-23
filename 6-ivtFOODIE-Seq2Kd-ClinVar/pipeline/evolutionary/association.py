#!/usr/bin/env python3
"""Unsupervised TF-adjusted correlations with TF cluster bootstrap and within-TF permutation."""
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from statsmodels.stats.multitest import multipletests
from shared import *
BOOT=2000;PERM=1999

def centered_ranks(d,cols,group='tf'):
 z=d[cols].rank(method='average').to_numpy(float)
 # Within-TF percentile ranks, centered; variation between TF distributions removed.
 for indices in d.groupby(group,sort=False).indices.values():
  for j,col in enumerate(cols):
   ranks=rankdata(d.iloc[indices][col].to_numpy());z[indices,j]=(ranks-ranks.mean())/len(indices)
 return z

def partial_from_cov(cov):
 if cov.shape[0]>2:
  a=cov[:2,:2];b=cov[:2,2:];cc=cov[2:,2:];v=a-b@np.linalg.pinv(cc)@b.T
 else:v=cov
 den=np.sqrt(max(0,v[0,0])*max(0,v[1,1]));return float(np.clip(v[0,1]/den,-1,1)) if den>1e-15 else np.nan

def association(d,x,y,controls=(),seed=42,permutations=True):
 cols=list(dict.fromkeys([x,y]+list(controls)));d=d.dropna(subset=cols).copy()
 # Multi-TF SNPs excluded from composition-controlled analysis; repeated selected fragments averaged first.
 d=d.groupby(['variant_id','tf'],as_index=False)[cols].mean();multi=d.groupby('variant_id').tf.nunique();d=d[~d.variant_id.isin(multi[multi>1].index)].reset_index(drop=True)
 ix=[np.asarray(v) for v in d.groupby('tf',sort=False).indices.values() if len(v)>=3];keep=np.concatenate(ix) if ix else np.array([],int);d=d.iloc[keep].reset_index(drop=True)
 groups=[np.asarray(v) for v in d.groupby('tf',sort=False).indices.values()];result=dict(n_variants=len(d),n_tfs=len(groups),controls=';'.join(controls))
 if len(d)<10 or len(groups)<3 or d[x].nunique()<2 or d[y].nunique()<2:return dict(result,rho=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status='insufficient_data')
 variable=d.groupby('tf')[cols].nunique();result.update(n_tfs_score_variable=int((variable[x]>1).sum()),n_tfs_effect_variable=int((variable[y]>1).sum()),n_tfs_both_variable=int(((variable[x]>1)&(variable[y]>1)).sum()))
 z=centered_ranks(d,cols);cov=z.T@z;rho=partial_from_cov(cov)
 if not np.isfinite(rho):return dict(result,rho=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,status='no_informative_within_TF_variation')
 rng=np.random.default_rng(seed)
 # Clusters have exact zero means; resampling TF clusters keeps the within-TF rank estimand.
 sums=np.array([z[g].T@z[g] for g in groups]);equal_sums=sums/np.array([len(g) for g in groups])[:,None,None];rho_equal=partial_from_cov(equal_sums.sum(0));boots=[];equal_boots=[]
 for _ in range(BOOT):
  indices=rng.integers(len(groups),size=len(groups));boots.append(partial_from_cov(sums[indices].sum(0)));equal_boots.append(partial_from_cov(equal_sums[indices].sum(0)))
 finite=np.asarray(boots)[np.isfinite(boots)];lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
 equal_finite=np.asarray(equal_boots)[np.isfinite(equal_boots)];elo,ehi=np.quantile(equal_finite,[.025,.975]) if len(equal_finite) else (np.nan,np.nan)
 pval=np.nan
 if permutations and np.isfinite(rho) and len(cols)==2:
  hit=0
  for _ in range(PERM):
   shuffled=z[:,0].copy()
   for g in groups:shuffled[g]=rng.permutation(shuffled[g])
   v=np.column_stack([shuffled,z[:,1:]]);rr=partial_from_cov(v.T@v);hit+=abs(rr)>=abs(rho)
  pval=(hit+1)/(PERM+1)
 return dict(result,rho=rho,ci_low=lo,ci_high=hi,rho_equal_TF=rho_equal,ci_low_equal_TF=elo,ci_high_equal_TF=ehi,p=pval,status='ok',bootstrap_replicates=len(finite),permutation_replicates=PERM if permutations and len(cols)==2 else 0)

def main():
 d=pd.read_parquet(OUT/'tables/association_input.parquet');rows=[];tfrows=[]
 metrics=['mean_abs_delta','sd_delta','dna_rank_loss','interaction_rms','robust_reversal_fraction','mean_delta']
 for (source,context,consequence),g in d.groupby(['source','context','consequence']):
  for score in ['evolutionary_score','am_score_validated']:
   if score=='am_score_validated' and (consequence!='missense' or context!='dbd'):continue
   for metric in metrics:
    endpoint=f'{source}.{context}.{consequence}.{score}.{metric}';res=association(g,score,metric,seed=int(digest(endpoint)[:8],16),permutations=True)
    rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='TF_adjusted',stratum='all',**res))
    # Pooled rho is descriptive: it can reflect TF composition.
    v=g.groupby('variant_id')[[score,metric]].mean().dropna();rr=spearmanr(v[score],v[metric]).statistic if len(v)>2 and v[score].nunique()>1 and v[metric].nunique()>1 else np.nan
    rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='pooled_descriptive',stratum='all',n_variants=len(v),n_tfs=g.tf.nunique(),rho=rr,status='descriptive_only'))
    controls=[]
    if metric in ('sd_delta','dna_rank_loss','interaction_rms','robust_reversal_fraction'):controls.append('mean_abs_delta')
    if consequence=='stop_gain':controls+=['deleted_residues','deleted_fraction']
    if controls:rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='TF_adjusted_partial',stratum='all',**association(g,score,metric,controls,seed=int(digest(endpoint+'partial')[:8],16),permutations=False)))
    if consequence=='missense' and metric in ['mean_abs_delta','sd_delta','dna_rank_loss','interaction_rms','robust_reversal_fraction']:
     uncropped=g[(g.fragment_end0-g.fragment_start0)<=512];controls=['mean_abs_delta'] if metric!='mean_abs_delta' else []
     rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='TF_adjusted_partial' if controls else 'TF_adjusted',stratum='uncropped_DBD',**association(uncropped,score,metric,controls,permutations=False)))
    if metric=='mean_abs_delta':
     for name,gg in [('stars_ge1',g[g.review_stars>=1]),('stars_ge2',g[g.review_stars>=2])]+[(label,g[g.label==label]) for label in LABELS]:
      rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='TF_adjusted',stratum=name,**association(gg,score,metric,seed=int(digest(endpoint+name)[:8],16),permutations=False)))
     for tf,gg in g.groupby('tf'):
      v=gg.groupby('variant_id')[[score,metric]].mean().dropna()
      if len(v)>=8 and v[score].nunique()>1 and v[metric].nunique()>1:tfrows.append(dict(source=source,context=context,consequence=consequence,score=score,tf=tf,n_variants=len(v),rho=spearmanr(v[score],v[metric]).statistic))
  # Matched ESM/AM comparison and incremental association, same SNPs and contexts.
  if consequence=='missense':
   matched=g.dropna(subset=['evolutionary_score','am_score_validated'])
   for metric in metrics:
    for score,other in [('am_score_validated','evolutionary_score'),('evolutionary_score','am_score_validated')]:
     rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='matched_TF_adjusted',stratum='AM_ESM_common',**association(matched,score,metric,permutations=False)))
     controls=[other]+(['mean_abs_delta'] if metric in ['sd_delta','dna_rank_loss','interaction_rms','robust_reversal_fraction'] else [])
     rows.append(dict(source=source,context=context,consequence=consequence,score=score,metric=metric,analysis='matched_partial',stratum='AM_ESM_common',**association(matched,score,metric,controls,permutations=False)))
   rows.append(dict(source=source,context=context,consequence=consequence,score='am_score_validated',metric='evolutionary_score',analysis='TF_adjusted',stratum='AM_ESM_common',**association(matched,'am_score_validated','evolutionary_score',permutations=False)))
  else:
   for metric in metrics:
    rows.append(dict(source=source,context=context,consequence=consequence,score='q05_deleterious_proxy',metric=metric,analysis='TF_adjusted_partial',stratum='truncation_q05_sensitivity',**association(g,'q05_deleterious_proxy',metric,['deleted_residues','deleted_fraction'],permutations=False)))
  print('Done associations',source,context,consequence,flush=True)
 table=pd.DataFrame(rows);table['rho_raw_LLR']=np.nan;table['ci_low_raw_LLR']=np.nan;table['ci_high_raw_LLR']=np.nan
 mask=(table.score=='evolutionary_score')&(table.consequence=='missense')
 table.loc[mask,'rho_raw_LLR']=-table.loc[mask,'rho'];table.loc[mask,'ci_low_raw_LLR']=-table.loc[mask,'ci_high'];table.loc[mask,'ci_high_raw_LLR']=-table.loc[mask,'ci_low']
 table['q']=np.nan;mask=table.p.notna();table.loc[mask,'q']=multipletests(table.loc[mask,'p'],method='fdr_bh')[1]
 table.to_csv(OUT/'tables/correlations.csv',index=False);pd.DataFrame(tfrows).to_csv(OUT/'tables/per_tf_correlations.csv',index=False)
 save_json(dict(bootstrap=BOOT,permutations=PERM,seed_policy='default42; primary/stratified endpoint seeds from first8 SHA256 hex digits; see code',primary_estimand='Pearson correlation of centered within-TF percentile ranks (TF-adjusted Spearman analogue)',ci='percentile TF cluster bootstrap, re-estimated partial covariance each draw',p='within-TF permutation only for unconditioned all-cohort endpoints; other endpoints use CI only',multiple_testing='BH across reported permutation p values',snp_unit='selected fragments averaged per SNP/TF; SNPs assigned to multiple TFs excluded from adjusted analysis',TF_min_variants=3,warning='shared ESM2 encoder; all associations observational model behavior'),OUT/'qc/statistical_methods.json')
if __name__=='__main__':main()
