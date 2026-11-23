#!/usr/bin/env python3
"""Complete AM-only branch while ESM masked queue continues; joint comparisons deferred."""
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shared import *
from association import association,centered_ranks
from statsmodels.stats.multitest import multipletests

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--render-only',action='store_true');args=ap.parse_args()
 setup();m=pd.read_parquet(OUT/'tables/seq2kd_dna_metrics.parquet');am=pd.read_parquet(OUT/'tables/alphamissense_mapping.parquet')[['mapping_id','am_score_validated','am_mapping_status']]
 d=m[m.consequence=='missense'].merge(am,on='mapping_id',validate='many_to_one');rows=[]
 if args.render_only:
  table=pd.read_csv(OUT/'tables/alphamissense_complete_correlations.csv')
 else:
  for source,g in d.groupby('source'):
   for metric in ['mean_abs_delta','sd_delta','dna_rank_loss','interaction_rms','mean_delta']:
    rows.append(dict(source=source,metric=metric,analysis='TF_adjusted',**association(g,'am_score_validated',metric,permutations=True)))
    if metric in ['sd_delta','dna_rank_loss','interaction_rms']:rows.append(dict(source=source,metric=metric,analysis='TF_adjusted_partial',**association(g,'am_score_validated',metric,['mean_abs_delta'],permutations=False)))
   print('AM-only source complete',source,flush=True)
  table=pd.DataFrame(rows);table['q']=np.nan;mask=table.p.notna();table.loc[mask,'q']=multipletests(table.loc[mask,'p'],method='fdr_bh')[1];table.to_csv(OUT/'tables/alphamissense_complete_correlations.csv',index=False)
 coverage=json.loads((OUT/'qc/alphamissense_coverage.json').read_text());lines=['# AlphaMissense全量分支（ESM联合结果待完成）','',f'有Seq2Kd预测的错义SNP {coverage["unique_snps"]}；一致映射 {coverage["consistent_unique_snps"]}；未一致映射 {coverage["nonconsistent_unique_snps"]}（无官方记录{coverage["absent_unique_snps"]}，有记录未核验一致{coverage["present_but_nonconsistent_unique_snps"]}）。','', '本文件的置换检验FDR家族仅限AM分支；联合分析完成后以correlations.csv中的联合家族q值为准。以下为TF内中心化百分位秩相关、按TF聚类bootstrap 95% CI。分层过滤/缺失指标会使相关分析样本量小于映射总数。','', '|DNA来源|指标|调整|SNP数|TF数|rho|95% CI|','|---|---|---|---:|---:|---:|---|']
 for r in table.itertuples():lines.append(f'|{r.source}|{r.metric}|{r.analysis}|{r.n_variants}|{r.n_tfs}|{r.rho:.3f}|[{r.ci_low:.3f}, {r.ci_high:.3f}]|')
 lines+=['','[逐SNP映射状态](tables/alphamissense_unique_snp_coverage.csv)；[逐映射核验](tables/alphamissense_mapping.csv)；[全部AM关联](tables/alphamissense_complete_correlations.csv)。','', '本分支尚未控制ESM分数，因此不能回答AlphaMissense是否提供独立补充信息。没有ClinVar监督训练。该相关是模型预测行为，不代表真实Kd或因果机制。条件相关CI为探索性逐项区间。']
 (OUT/'AlphaMissense_全量分支.md').write_text('\n'.join(lines)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(10,4));plt.rcParams['pdf.fonttype']=42
 for ax,(source,g) in zip(axes,d.groupby('source')):
  v=g.dropna(subset=['am_score_validated','mean_abs_delta']).groupby(['variant_id','tf'],as_index=False)[['am_score_validated','mean_abs_delta']].mean();v=v.groupby('tf').filter(lambda x:len(x)>=3).reset_index(drop=True);z=centered_ranks(v,['am_score_validated','mean_abs_delta']);hb=ax.hexbin(z[:,0],z[:,1],gridsize=35,mincnt=1,cmap='Blues',linewidths=0);fig.colorbar(hb,ax=ax,label='SNP count / hexagon');ax.set_xlabel('Within-TF AM score percentile (centered)');ax.set_ylabel('Within-TF mean |delta pKd| percentile (centered)');ax.set_title(f'{source}: n={len(v):,} SNPs')
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'alphamissense_complete.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig);save_json(dict(AM_branch_complete=True,ESM_conditional_analysis_complete=False),OUT/'qc/AM_BRANCH_COMPLETE.json')
if __name__=='__main__':main()
