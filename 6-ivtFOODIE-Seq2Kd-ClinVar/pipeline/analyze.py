#!/usr/bin/env python3
"""Summarize verified ClinVar predictions and plot strict clinical comparisons."""
import argparse,json,itertools,collections
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
def p_text(value):return 'underflow' if value==0 else f'{value:.3g}'
COLORS={'Benign':'#4477AA','Likely_benign':'#88CCEE','Likely_pathogenic':'#CCBB44','Pathogenic':'#AA3377'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
OUT=RUN/'analysis';FIG=OUT/'figures';TABLE=OUT/'tables';TESTS=[];ENDPOINTS=[];SUMMARIES=[];METRICS=[];TF_UNAVAILABLE=[]

def savefig(fig,name):
 FIG.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)

def compare(d,endpoint,value,unit,title,primary=True):
 d=d.dropna(subset=[value,'label']).copy()
 if d.variant_id.duplicated().any():raise ValueError('Nonunique variant statistics: '+endpoint)
 parquet(d,OUT/'variant_summaries'/f'{endpoint}.parquet')
 if d.empty:
  save_json({'endpoint':endpoint,'skipped':'no observations'},OUT/'variant_summaries'/f'{endpoint}.json');return d
 labels_to_plot=['Benign','Pathogenic']
 groups={lab:d.loc[d.label==lab,value].to_numpy() for lab in labels_to_plot}
 available=[x for x in groups.values() if len(x)]
 overall=1.0
 if len(available)>1 and np.ptp(np.concatenate(available))>0:overall=float(stats.kruskal(*available).pvalue)
 if len(available)>1:TESTS.append(dict(endpoint=endpoint,test='Kruskal-Wallis',class_a='all',class_b='all',p=overall,n=len(d),effect=np.nan,primary=primary))
 for a,b in itertools.combinations(labels_to_plot,2):
  x,y=groups[a],groups[b]
  if not len(x) or not len(y):continue
  result=stats.mannwhitneyu(x,y,alternative='two-sided',method='auto')
  TESTS.append(dict(endpoint=endpoint,test='Mann-Whitney U',class_a=a,class_b=b,p=float(result.pvalue),n=len(x)+len(y),n_a=len(x),n_b=len(y),effect=2*float(result.statistic)/(len(x)*len(y))-1,primary=primary))
 for lab,values in groups.items():
  if len(values):SUMMARIES.append(dict(endpoint=endpoint,label=lab,n=len(values),median=float(np.median(values)),q25=float(np.percentile(values,25)),q75=float(np.percentile(values,75)),mean=float(values.mean()),unit=unit,primary=primary))
 ENDPOINTS.append(dict(endpoint=endpoint,unit=unit,title=title,primary=primary,n=len(d),kw_p=overall,kw_estimable=len(available)>1))
 # Main scientific figures: real distributions, zero-preserving symlog, no sampled p-values.
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),gridspec_kw={'width_ratios':[1,1.25]})
 pos=[];vals=[];labels=[]
 for i,lab in enumerate(labels_to_plot):
  a=groups[lab]
  if len(a):pos.append(i);vals.append(a);labels.append(lab)
 if vals:
  box=axes[0].boxplot(vals,positions=pos,widths=.55,showfliers=False,patch_artist=True,medianprops={'color':'#333333','linewidth':1.2})
  for patch,lab in zip(box['boxes'],labels):patch.set_facecolor(COLORS[lab]);patch.set_edgecolor('#333333')
 for i,lab in enumerate(labels_to_plot):
  a=groups[lab]
  if len(a):
   axes[1].step(np.sort(a),np.arange(1,len(a)+1)/len(a),where='post',marker='o' if len(a)==1 else None,label=f'{lab.replace("_"," ")} (n={len(a):,})',color=COLORS[lab],lw=1.8,ls=['-','--','-.',':'][i])
 axes[0].set_xticks(range(len(labels_to_plot)));axes[0].set_xticklabels([x.replace('_','\n')+f'\nn={len(groups[x]):,}' for x in labels_to_plot]);axes[0].set_ylabel(unit)
 if value!='tf_adjusted_rank':axes[0].set_yscale('symlog',linthresh=.001);axes[1].set_xscale('symlog',linthresh=.001)
 if len(d) and d[value].min()>=0:axes[0].set_ylim(bottom=0);axes[1].set_xlim(left=0)
 axes[1].set_xlabel(unit);axes[1].set_ylabel('Cumulative fraction');axes[1].legend(frameon=False,fontsize=8,loc='lower right')
 axes[0].grid(axis='y',alpha=.2);axes[1].grid(alpha=.2)
 fig.suptitle(title);fig.text(.01,-.04,'One observation per unique SNV. Box: median/IQR; whiskers: 1.5 IQR. Main scales are symlog (linear near zero).',fontsize=8)
 fig.tight_layout();savefig(fig,endpoint)
 if primary and d.label.nunique()>1:
  strict=d[d.label.isin(labels_to_plot)]
  y=strict.label.eq('Pathogenic').astype(int).to_numpy();pred=strict[value].to_numpy()
  if len(np.unique(y))==2:
   roc=roc_auc_score(y,pred);ap=average_precision_score(y,pred);METRICS.append(dict(endpoint=endpoint,roc_auc=roc,pr_auc_average_precision=ap,positive_prevalence=float(y.mean()),n=len(strict)))
   fpr,tpr,_=roc_curve(y,pred);precision,recall,_=precision_recall_curve(y,pred)
   fig,ax=plt.subplots(1,2,figsize=(9,3.7));ax[0].plot(fpr,tpr,color='#4477AA',label=f'AUC={roc:.3f}');ax[0].plot([0,1],[0,1],color='grey',ls='--');ax[0].set(xlabel='False positive rate',ylabel='True positive rate')
   ax[1].plot(recall,precision,color='#AA3377',label=f'AP={ap:.3f}');ax[1].axhline(y.mean(),color='grey',ls='--');ax[1].set(xlabel='Recall',ylabel='Precision')
   for a in ax:a.legend(frameon=False);a.set_xlim(0,1);a.set_ylim(0,1)
   fig.suptitle(title+' — Pathogenic vs Benign');fig.tight_layout();savefig(fig,endpoint+'.roc_pr')
 return d



def summarize_coding():
    annotation = pd.read_parquet(RUN / 'prepared/coding_variants.parquet')
    shards = sorted((RUN / 'scores').glob('coding.*.parquet'))
    scores = pd.concat([pd.read_parquet(path) for path in shards if not path.stem.endswith('.raw')])
    scores = scores.loc[scores.source.eq('streme') & scores.n_dna.eq(100)]
    scores = scores.merge(annotation[['mapping_id', 'label', 'consequence', 'review_stars']],
                          on='mapping_id', validate='one_to_one')
    for consequence in ('all', 'missense', 'stop_gain', 'stop_loss'):
        selected = scores if consequence == 'all' else scores[scores.consequence.eq(consequence)]
        if selected.empty:
            continue
        summary = selected.groupby('variant_id', as_index=False).agg(
            value=('mean_abs_delta', 'mean'), label=('label', 'first'),
            review_stars=('review_stars', 'first'), tf=('tf', lambda tf: ';'.join(sorted(set(tf)))))
        compare(summary, f'coding.streme.{consequence}.n100', 'value', 'Mean absolute ΔpKd',
                f'Coding · STREME · {consequence} · 100 DNA sequences')


def summarize_promoter(percentages, include_probound):
    annotations = pd.read_parquet(RUN / 'prepared/promoter_variants.parquet')
    scores = pd.concat([pd.read_parquet(path) for path in sorted((RUN / 'scores').glob('promoter.*.parquet'))])
    methods = {'seq2kd_streme': ('streme', 'max', 'Mean absolute ΔpKd')}
    if include_probound:
        methods['probound'] = ('probound', 'median', 'Mean absolute log10 relative Ka ratio')
    for method, (source, combine_features, unit) in methods.items():
        candidates = pd.concat([pd.read_parquet(path) for path in sorted((RUN / 'candidates' / source).glob('*.parquet'))])
        if source == 'streme':
            candidates = candidates.merge(scores[['variant_id', 'protein_id', 'ref_pkd', 'alt_pkd']],
                                          on=['variant_id', 'protein_id'], how='left', validate='one_to_one')
            if candidates[['ref_pkd', 'alt_pkd']].isna().any().any():
                raise ValueError('Incomplete STREME candidate coverage')
            candidates['delta'] = candidates.alt_pkd - candidates.ref_pkd
            candidates['abs_delta'] = candidates.delta.abs()
        parquet(candidates, OUT / 'promoter_pair_scores' / f'{method}.parquet')
        for pct in percentages:
            kept = candidates[candidates[f'top{pct}']]
            by_tf = kept.groupby(['variant_id', 'tf'], as_index=False).agg(value=('abs_delta', combine_features))
            summary = by_tf.groupby('variant_id', as_index=False).agg(mean=('value', 'mean'), n_tf=('tf', 'size'))
            summary = summary.merge(annotations[['variant_id', 'label', 'review_stars']],
                                    on='variant_id', validate='one_to_one')
            compare(summary, f'promoter.{method}.top{pct}.mean', 'mean', unit,
                    f'Promoter · {method} · top {pct}%', primary=method == 'seq2kd_streme' and pct == 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threshold-sensitivity', action='store_true')
    parser.add_argument('--include-probound', action='store_true')
    args = parser.parse_args()
    from verify_outputs import verify
    verify()
    TABLE.mkdir(parents=True, exist_ok=True)
    summarize_coding()
    summarize_promoter(PCTS if args.threshold_sensitivity else [5], args.include_probound)
    tests = pd.DataFrame(TESTS)
    tests.to_csv(TABLE / 'statistical_tests.csv', index=False)
    pd.DataFrame(SUMMARIES).to_csv(TABLE / 'group_distributions.csv', index=False)
    pd.DataFrame(METRICS).to_csv(TABLE / 'discrimination.csv', index=False)
    save_json(ENDPOINTS, OUT / 'figure_index.json')
    save_json({'complete': True, 'endpoints': len(ENDPOINTS), 'labels': ['Benign', 'Pathogenic'],
               'analysis_script_sha256': filehash(__file__)}, RUN / 'FINAL_COMPLETE.json')


if __name__ == '__main__':
    main()
