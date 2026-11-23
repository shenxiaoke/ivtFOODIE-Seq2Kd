"""Coverage denominators by TF, clinical label and review stars, preserving unique SNP units."""
import pandas as pd
from shared import *
def summarize():
 a=pd.read_parquet(OUT/'tables/alphamissense_mapping.parquet')
 a['validated']=a.am_mapping_status.eq('consistent');a['official_present']=a.am_score.notna()
 # Within each TF, count each SNP once; global clinical/star counts deduplicate across TFs.
 tf=a.groupby(['tf','variant_id','label','review_stars'],as_index=False).agg(validated=('validated','any'),official_present=('official_present','any'))
 rows=[]
 for (name,label),g in tf.groupby(['tf','label']):rows.append(dict(tf=name,label=label,n_snps=len(g),validated=int(g.validated.sum()),not_validated=int((~g.validated).sum()),official_absent=int((~g.official_present).sum())))
 pd.DataFrame(rows).to_csv(OUT/'tables/AM_coverage_by_TF_and_label.csv',index=False)
 unique=a.groupby(['variant_id','label','review_stars'],as_index=False).agg(validated=('validated','any'),official_present=('official_present','any'))
 rows=[]
 for by in ['label','review_stars']:
  for name,g in unique.groupby(by):rows.append(dict(stratum=by,value=name,n_snps=len(g),validated=int(g.validated.sum()),not_validated=int((~g.validated).sum()),coverage=float(g.validated.mean())))
 pd.DataFrame(rows).to_csv(OUT/'tables/AM_coverage_by_clinical_and_stars.csv',index=False)
if __name__=='__main__':summarize()
