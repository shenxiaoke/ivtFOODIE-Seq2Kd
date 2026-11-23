#!/usr/bin/env python3
"""Genome-first official AM join; verify GENCODE v32 translation and exact genomic codon."""
import gzip,re,hashlib,time
import pandas as pd,numpy as np,pysam
from Bio.Seq import Seq
from shared import *

def translate_tx(t,genome):
 parts=sorted(t['cds'],reverse=t['strand']=='-');dna='';regions=[]
 for a,b,phase in parts:
  if phase!=(3-len(dna)%3)%3:return None,'phase_inconsistent_or_partial'
  s=genome.fetch(t['chrom'],a-1,b).upper();s=rc(s) if t['strand']=='-' else s
  regions.append((a,b,len(dna)));dna+=s
 if len(dna)%3:return None,'incomplete_cds'
 prot=str(Seq(dna).translate())
 # GFF3 versions can include the terminal stop; remove only a terminal stop.
 if prot.endswith('*'):prot=prot[:-1]
 if '*' in prot:return None,'internal_stop'
 return (dna,prot,regions),'ok'

def main():
 setup();c=predicted_coding();c=c[c.consequence=='missense'].copy();f=DB/'AlphaMissense_hg38.tsv.gz'
 if Path(str(f)+'.aria2').exists():raise RuntimeError('AM hg38 download is not complete')
 keys={(r.chrom,str(r.pos1),r.ref,r.alt) for r in c.itertuples()};hits=[];h=hashlib.md5();n=0
 # Full decompression validates gzip CRC and EOF as well as parsing all 71M official variants.
 with gzip.open(f,'rt') as stream:
  for line in stream:
   if line.startswith('#'):continue
   a=line.rstrip('\n').split('\t');n+=1
   if (a[0],a[1],a[2],a[3]) in keys:hits.append(a)
   if n%10000000==0:print('Official rows scanned',n,'matched rows',len(hits),flush=True)
 cols=['am_chrom','am_pos1','am_ref','am_alt','am_genome','am_uniprot','am_transcript','am_protein_variant','am_score','am_class']
 am=pd.DataFrame(hits,columns=cols);am['am_pos1']=am.am_pos1.astype(int);am['am_score']=am.am_score.astype(float)
 if not am.am_score.between(0,1).all():raise ValueError('Invalid AM scores')
 parquet(am,OUT/'tables/alphamissense_genomic_hits.parquet')
 tids=set(am.am_transcript);tx={};gff=DB/'gencode.v32.annotation.gff3.gz'
 with gzip.open(gff,'rt') as stream:
  for line in stream:
   if line.startswith('#'):continue
   a=line.rstrip().split('\t')
   if a[2] not in ('transcript','CDS'):continue
   m=re.search(r'(?:^|;)transcript_id=([^;]+)',a[8]);tid=m[1] if m else ''
   if tid not in tids:continue
   attrs=dict(x.split('=',1) for x in a[8].split(';') if '=' in x)
   t=tx.setdefault(tid,dict(chrom=a[0],strand=a[6],gene_id=attrs['gene_id'].split('.')[0],cds=[]))
   if a[2]=='CDS':t['cds'].append((int(a[3]),int(a[4]),int(a[7])))
 # CDS identity can establish a gene-annotation bridge when stable gene IDs changed.
 current_tx=set(c.transcript_id);v48={}
 with open(SHARE/'clinvar/gencode.v48.basic.annotation.gtf') as stream:
  for line in stream:
   if line.startswith('#'):continue
   a=line.rstrip().split('\t')
   if a[2]!='CDS':continue
   m=re.search(r'transcript_id "([^"]+)"',a[8]);tid=m[1] if m else ''
   if tid in current_tx:
    t=v48.setdefault(tid,dict(chrom=a[0],strand=a[6],cds=[]));t['cds'].append((int(a[3]),int(a[4]),int(a[7])))
 def coding_path(t,n_bases):
  path=[];remaining=n_bases
  for a,b,phase in sorted(t['cds'],reverse=t['strand']=='-'):
   n=min(b-a+1,remaining)
   if n<=0:break
   path.append((a,a+n-1) if t['strand']=='+' else (b-n+1,b));remaining-=n
  return (t['chrom'],t['strand'],tuple(path)) if remaining==0 else None
 genome=pysam.FastaFile(str(SHARE/'clinvar/hg38.fa'));translated={};taudit=[]
 for tid,t in tx.items():
  result,status=translate_tx(t,genome);translated[tid]=(result,status)
  taudit.append(dict(transcript_id=tid,status=status,gene_id=t['gene_id'],sequence_sha256=digest(result[1]) if result else '',length=len(result[1]) if result else 0))
 pd.DataFrame(taudit).to_csv(OUT/'qc/am_v32_transcript_translation.csv',index=False)
 proteins=pd.read_parquet(RUN/'prepared/proteins.parquet').set_index('protein_id')
 merged=c.merge(am,left_on=['chrom','pos1','ref','alt'],right_on=['am_chrom','am_pos1','am_ref','am_alt'],how='left',validate='many_to_many')
 statuses=[];proofs=[]
 for r in merged.itertuples():
  status='consistent';proof={}
  if pd.isna(r.am_score):status='no_official_genomic_record'
  elif r.am_genome!='hg38':status='genome_build_mismatch'
  elif r.am_uniprot!=r.uniprot:status='uniprot_mismatch'
  elif r.am_protein_variant!=f'{r.ref_aa}{r.aa_pos1}{r.alt_aa}':status='amino_acid_change_mismatch'
  elif r.am_transcript not in translated:status='v32_transcript_missing'
  else:
   result,reason=translated[r.am_transcript];t=tx[r.am_transcript]
   if not result:status='v32_translation_'+reason
   else:
    dna,prot,regions=result;full=proteins.loc[r.protein_id,'full_sequence']
    proof=dict(v32_protein_sha256=digest(prot),current_protein_sha256=digest(full),same_transcript_version=r.transcript_id==r.am_transcript)
    gene_same=t['gene_id']==r.gene_id
    cds_same=(r.transcript_id in v48 and coding_path(t,len(prot)*3)==coding_path(v48[r.transcript_id],len(prot)*3))
    proof.update(v32_gene_id=t['gene_id'],same_gene_id=gene_same,identical_CDS_path=cds_same,gene_mapping_rule='same_gene_id' if gene_same else 'identical_CDS_and_full_protein' if cds_same and prot==full else 'unverified_gene_change')
    if prot!=full:status='full_reference_sequence_mismatch'
    elif not gene_same and not cds_same:status='gene_mismatch'
    else:
     offsets=[cum+(r.pos1-a if t['strand']=='+' else b-r.pos1) for a,b,cum in regions if a<=r.pos1<=b]
     if len(offsets)!=1:status='v32_variant_not_unique_CDS'
     else:
      off=offsets[0];aa0=off//3;ref=r.ref if t['strand']=='+' else rc(r.ref);alt=r.alt if t['strand']=='+' else rc(r.alt)
      codon=dna[aa0*3:aa0*3+3];mut=codon[:off%3]+alt+codon[off%3+1:]
      if dna[off]!=ref or aa0+1!=r.aa_pos1 or str(Seq(codon).translate())!=r.ref_aa or str(Seq(mut).translate())!=r.alt_aa:status='v32_genomic_to_protein_mismatch'
  statuses.append(status);proofs.append(proof)
 merged['am_mapping_status']=statuses
 merged=pd.concat([merged.reset_index(drop=True),pd.DataFrame(proofs)],axis=1)
 # The official genomic table can contain several transcripts for one allele.
 # Verify every record, retain all evidence; select one identical-score consistent match.
 parquet(merged.drop(columns=['wt_sequence','alt_sequence']),OUT/'tables/alphamissense_all_match_evidence.parquet')
 resolved=[]
 for mid,g in merged.groupby('mapping_id',sort=False):
  good=g[g.am_mapping_status=='consistent']
  if len(good):
   row=good.sort_values(['same_transcript_version','am_transcript'],ascending=[False,True]).iloc[0].copy()
   if good.am_score.nunique()!=1:row['am_mapping_status']='ambiguous_consistent_official_scores'
  else:row=g.sort_values('am_mapping_status').iloc[0].copy()
  row['official_match_records']=int(g.am_score.notna().sum());row['all_match_statuses']=';'.join(sorted(set(g.am_mapping_status)));resolved.append(row)
 merged=pd.DataFrame(resolved).reset_index(drop=True)
 merged['am_score_validated']=merged.am_score.where(merged.am_mapping_status.eq('consistent'))
 keep=[x for x in merged if x not in ('wt_sequence','alt_sequence')]
 parquet(merged[keep],OUT/'tables/alphamissense_mapping.parquet');merged[keep].to_csv(OUT/'tables/alphamissense_mapping.csv',index=False)
 # Unique SNPs can have multiple selected fragments. Retain status-set to avoid double counting.
 unique=merged.groupby('variant_id').agg(mapping_statuses=('am_mapping_status',lambda x:';'.join(sorted(set(x)))),consistent=('am_mapping_status',lambda x:(x=='consistent').any()),official_present=('am_score',lambda x:x.notna().any())).reset_index()
 unique.to_csv(OUT/'tables/alphamissense_unique_snp_coverage.csv',index=False)
 summary=dict(official_rows_scanned=n,predicted_missense_mappings=len(c),unique_snps=len(unique),consistent_unique_snps=int(unique.consistent.sum()),nonconsistent_unique_snps=int((~unique.consistent).sum()),absent_unique_snps=int((~unique.official_present).sum()),present_but_nonconsistent_unique_snps=int((unique.official_present&~unique.consistent).sum()),mapping_status_counts=merged.am_mapping_status.value_counts().to_dict(),unique_failure_reason_counts=unique.loc[~unique.consistent,'mapping_statuses'].value_counts().to_dict(),gene_id_changed_but_identical_CDS_and_protein=int(((merged.am_mapping_status=='consistent')&(merged.gene_mapping_rule=='identical_CDS_and_full_protein')).sum()),different_transcript_version_but_consistent=int(((merged.am_mapping_status=='consistent')&~merged.same_transcript_version.fillna(False).astype(bool)).sum()),inputs={str(p):filehash(p) for p in [f,gff,SHARE/'clinvar/gencode.v48.basic.annotation.gtf',RUN/'prepared/coding_variants.parquet',RUN/'prepared/proteins.parquet']})
 save_json(summary,OUT/'qc/alphamissense_coverage.json');from am_coverage import summarize;summarize();print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
