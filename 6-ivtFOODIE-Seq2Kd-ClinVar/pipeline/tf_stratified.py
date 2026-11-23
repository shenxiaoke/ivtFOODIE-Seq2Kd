"""Within-TF rank-sum sensitivity, excluding TF strata without both clinical classes."""
import itertools,math
import numpy as np
from scipy import stats
from common import seeded_rng

def stratified_rank_sum(frame,class_a,class_b,key,resamples=9999):
 d=frame[frame.label.isin([class_a,class_b])].copy()
 # Prevent one SNP contributing to independent TF strata more than once.
 counts=d.groupby('variant_id').tf.nunique();ambiguous=set(counts[counts>1].index);d=d[~d.variant_id.isin(ambiguous)]
 if d.duplicated(['variant_id','tf']).any():raise ValueError('Duplicate SNP within TF stratum')
 blocks=[];observed=0.;variance=0.;denominator=0.;na=0;nb=0;informative=0
 for tf,g in d.groupby('tf',sort=True):
  x=g.loc[g.label==class_a,'value'].to_numpy();y=g.loc[g.label==class_b,'value'].to_numpy()
  if not len(x) or not len(y):continue
  a,b=len(x),len(y);n=a+b;ranks=stats.rankdata(np.r_[x,y]);u=float(ranks[:a].sum()-a*(a+1)/2);mu=a*b/2
  _,ties=np.unique(np.r_[x,y],return_counts=True);tie_term=float((ties.astype(float)**3-ties).sum())
  var=a*b/12*((n+1)-tie_term/(n*(n-1)));w=1/(n+1)
  informative+=var>0
  observed+=w*(u-mu);variance+=w*w*var;denominator+=w*a*b;na+=a;nb+=b
  blocks.append((ranks,a,b,w,mu))
 if not blocks:return None
 effect=2*observed/denominator
 if variance<=0:return dict(p=1.,effect=effect,n_a=na,n_b=nb,n=na+nb,n_tf=len(blocks),n_informative_tf=int(informative),p_method='constant_scores',resamples=0,excluded_ambiguous_snvs=len(ambiguous))
 if informative>=10:
  p=float(2*stats.norm.sf(abs(observed)/np.sqrt(variance)));method='normal_tie_corrected';draws=0
 else:
  combinations=1
  for ranks,a,b,w,mu in blocks:
   combinations*=math.comb(len(ranks),min(a,b))
   if combinations>20000:break
  if combinations<=20000:
   null=np.zeros(1)
   for ranks,a,b,w,mu in blocks:
    k=min(a,b);offset=k*(k+1)/2;values=[]
    for indices in itertools.combinations(range(len(ranks)),k):
     small_u=float(ranks[list(indices)].sum()-offset);u=small_u if a<=b else a*b-small_u;values.append(w*(u-mu))
    null=(null[:,None]+np.array(values)[None,:]).ravel()
   p=float(np.mean(abs(null)>=abs(observed)-1e-10));method='exact_within_TF_permutation';draws=len(null)
  else:
   rng=seeded_rng('tf-stratified:'+key+':'+class_a+':'+class_b);exceed=0
   for draw in range(resamples):
    total=0.
    for ranks,a,b,w,mu in blocks:
     k=min(a,b);small_u=float(ranks[rng.choice(len(ranks),k,replace=False)].sum()-k*(k+1)/2);u=small_u if a<=b else a*b-small_u;total+=w*(u-mu)
    exceed+=abs(total)>=abs(observed)-1e-10
   p=(exceed+1)/(resamples+1);method='Monte_Carlo_within_TF_permutation';draws=resamples
 return dict(p=p,effect=effect,n_a=na,n_b=nb,n=na+nb,n_tf=len(blocks),n_informative_tf=int(informative),p_method=method,resamples=draws,excluded_ambiguous_snvs=len(ambiguous))
