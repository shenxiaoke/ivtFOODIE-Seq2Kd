#!/usr/bin/env python3
"""ClinVar SNV -> strand/phase-validated CDS -> sequence-matched DBD mutation; gene promoters."""
import re,bisect,collections,time
import pandas as pd
import numpy as np
import pysam
from Bio.Seq import Seq
from common import *
from motifs import load_motifs,sample_dna

def stars(status):
 if status=='practice_guideline':return 4
 if status=='reviewed_by_expert_panel':return 3
 if status=='criteria_provided,_multiple_submitters,_no_conflicts':return 2
 if status.startswith('criteria_provided'):return 1
 return 0

def read_variants(genome):
 counts=collections.Counter();rows=[];ref_rejections=[];missing_rejections=[]
 with open(SHARE/'clinvar/clinvar_20250623.vcf') as f:
  for line in f:
   if line.startswith('##reference=') and not any(build in line for build in ['GRCh38','hg38']):raise ValueError('VCF reference is not GRCh38/hg38')
   if line.startswith('#'):continue
   counts['vcf_records']+=1
   chrom,pos,vid,ref,alt,qual,flt,info=line.rstrip().split('\t')[:8]
   if len(ref)!=1 or len(alt)!=1 or ref not in 'ACGT' or alt not in 'ACGT' or ref==alt:
    counts['not_biallelic_snv']+=1;continue
   d=dict(x.split('=',1) for x in info.split(';') if '=' in x)
   label=d.get('CLNSIG','')
   if label not in LABELS:counts['excluded_label:'+label]+=1;continue
   vcf_chrom=chrom
   chrom=normalize_chromosome(chrom);pos=int(pos)
   if chrom not in genome.references:
    counts['missing_contig']+=1;missing_rejections.append(dict(vcf_contig=vcf_chrom,attempted_fasta_contig=chrom,pos1=pos,ref=ref,alt=alt,variation_id=vid,label=label));continue
   if genome.fetch(chrom,pos-1,pos).upper()!=ref:
    counts['ref_mismatch']+=1;ref_rejections.append(dict(chrom=chrom,pos1=pos,ref=ref,alt=alt,fasta_ref=genome.fetch(chrom,pos-1,pos).upper(),variation_id=vid,label=label));continue
   status=d.get('CLNREVSTAT','')
   rows.append((f'{chrom}:{pos}:{ref}>{alt}',chrom,pos,vid,ref,alt,label,status,stars(status),d.get('ALLELEID','')))
 v=pd.DataFrame(rows,columns=['variant_id','chrom','pos1','variation_id','ref','alt','label','review_status','review_stars','allele_id'])
 # A variant must never enter two conflicting clinical groups.
 conflict=v.groupby('variant_id').label.nunique();bad=set(conflict[conflict>1].index)
 counts['duplicate_conflicting_variants']=len(bad)
 v=v[~v.variant_id.isin(bad)].sort_values(['chrom','pos1','variant_id']).drop_duplicates('variant_id').reset_index(drop=True)
 pd.DataFrame(missing_rejections,columns=['vcf_contig','attempted_fasta_contig','pos1','ref','alt','variation_id','label']).to_csv(RUN/'qc/missing_reference_contigs.csv',index=False)
 pd.DataFrame(ref_rejections,columns=['chrom','pos1','ref','alt','fasta_ref','variation_id','label']).to_csv(RUN/'qc/reference_mismatches.csv',index=False)
 counts['retained_snvs']=len(v);save_json(dict(counts),RUN/'qc/vcf_filter_counts.json');parquet(v,RUN/'prepared/variants.parquet');return v

def read_proteins():
 tf=pd.read_csv(SHARE/'clinvar/human_tf_final2_1636.csv').fillna('')
 tf=tf.set_index('uniprot');full=pd.read_csv(TFDIR/'human_tf_1636_full_seq.csv').set_index('uniprot').sequence.to_dict()
 rows=[];excluded=[]
 for fname in ['human_tf_part1_1545.csv','human_tf_part2_concat.csv','human_tf_part2_split.csv','human_tf_part3_83.csv']:
  for r in pd.read_csv(TFDIR/'junchen_new_20250426'/fname).fillna('').to_dict('records'):
   pid=r['uniprot'];uid=pid.split('.')[0];seq=r['sequence'].upper();reason=''
   if uid not in tf.index:reason='not_in_1636_TF_set'
   elif uid not in full:reason='missing_full_reference'
   elif any(x not in AA for x in seq):reason='noncanonical_or_discontinuous_sequence'
   else:
    wt=full[uid];start=wt.find(seq)
    if start<0:reason='fragment_not_exact_substring_of_reference'
    elif wt.find(seq,start+1)>=0:reason='ambiguous_fragment_mapping'
   if reason:excluded.append(dict(protein_id=pid,source_table=fname,reason=reason));continue
   meta=tf.loc[uid];geneid=str(meta.gene_id).split('.')[0]
   crop,crop0=centered_crop(seq)
   rows.append(dict(protein_id=pid,uniprot=uid,tf=meta.protein_name,gene_id=geneid,sequence=seq,
      full_sequence=wt,entry_name=str(meta['Entry Name']).split('_')[0],fragment_start0=start,fragment_end0=start+len(seq),source_table=fname,
      promoter_sequence=crop,promoter_crop_start0=crop0))
 p=pd.DataFrame(rows)
 if p.protein_id.duplicated().any():raise ValueError('Duplicate protein key')
 parquet(p,RUN/'prepared/proteins.parquet');pd.DataFrame(excluded).to_csv(RUN/'qc/protein_exclusions.csv',index=False);return p

def attrs(s):return dict(re.findall(r'(\w+) "([^"]*)"',s))
def gtf_data(target_genes):
 genes=[];tx={}
 with open(SHARE/'clinvar/gencode.v48.basic.annotation.gtf') as f:
  for line in f:
   if line.startswith('#'):continue
   c,src,feature,start,end,score,strand,phase,att=line.rstrip().split('\t');start=int(start);end=int(end)
   # Avoid parsing attributes on the vast majority of irrelevant exon/UTR records.
   if feature not in ('gene','transcript','CDS','stop_codon'):continue
   if feature=='gene':
    a=attrs(att)
    if a.get('gene_type')=='protein_coding':
     tss=start-1 if strand=='+' else end-1
     genes.append(dict(chrom=c,start0=max(0,tss-500),end0=tss+500,tss0=tss,gene_id=a['gene_id'].split('.')[0],gene_name=a['gene_name'],strand=strand))
    continue
   gid=re.search(r'gene_id "([^"]+)"',att)[1].split('.')[0]
   if gid not in target_genes:continue
   a=attrs(att);tid=a['transcript_id']
   t=tx.setdefault(tid,dict(transcript_id=tid,gene_id=gid,chrom=c,strand=strand,cds=[],stop=[],priority=3,tags=''))
   if feature=='transcript':
    t['tags']=';'.join(re.findall(r'tag "([^"]+)"',att));t['priority']=0 if 'MANE_Select' in att else 1 if 'Ensembl_canonical' in att else 2
   elif feature=='CDS':t['cds'].append((start,end,int(phase)))
   else:t['stop'].append((start,end))
 return pd.DataFrame(genes),tx

def transcript_cds(t,genome):
 segments=sorted(t['cds'],reverse=t['strand']=='-');sequence=[];cum=0;mapping=[]
 for a,b,phase in segments:
  if phase!=(3-cum%3)%3:return None,'phase_inconsistent_or_partial'
  s=genome.fetch(t['chrom'],a-1,b).upper();s=rc(s) if t['strand']=='-' else s
  sequence.append(s);mapping.append((a,b,cum));cum+=len(s)
 dna=''.join(sequence)
 if len(dna)%3 or not dna:return None,'incomplete_cds'
 prot=str(Seq(dna).translate())
 if '*' in prot:return None,'internal_stop'
 return (dna,prot,mapping),''

def promoter_variants(v,genes,genome):
 # Query variants once per gene; retain unique variants separately from all gene associations.
 bychr={c:(g.pos1.to_numpy(),g.index.to_numpy()) for c,g in v.groupby('chrom',sort=False)};pairs=[]
 for g in genes.itertuples():
  if g.chrom not in bychr:continue
  pos,ix=bychr[g.chrom];a=np.searchsorted(pos,g.start0+1);b=np.searchsorted(pos,g.end0,side='right')
  for i in ix[a:b]:pairs.append((v.at[i,'variant_id'],g.gene_id,g.gene_name,g.strand,g.tss0))
 links=pd.DataFrame(pairs,columns=['variant_id','promoter_gene_id','promoter_gene_name','promoter_strand','tss0']).drop_duplicates()
 subset=v[v.variant_id.isin(links.variant_id)].copy();rs=[];alts=[];valid=[]
 for r in subset.itertuples():
  s=genome.fetch(r.chrom,max(0,r.pos1-21),r.pos1+19).upper()
  ok=len(s)==40 and set(s)<=set('ACGT') and s[20]==r.ref
  valid.append(ok);rs.append(s);alts.append(s[:20]+r.alt+s[21:])
 subset['ref_seq']=rs;subset['alt_seq']=alts;subset=subset.loc[valid].reset_index(drop=True)
 links=links[links.variant_id.isin(subset.variant_id)]
 return subset,links

def coding_variants(v,proteins,tx,genome):
 bychr={c:(g.pos1.to_numpy(),g.index.to_numpy()) for c,g in v.groupby('chrom',sort=False)}
 by_gene={g:list(d.to_dict('records')) for g,d in proteins.groupby('gene_id')};rows=[];txaudit=[]
 for tid,t in sorted(tx.items()):
  if not t['cds'] or t['chrom'] not in bychr:continue
  result,error=transcript_cds(t,genome)
  if error:txaudit.append(dict(transcript_id=tid,gene_id=t['gene_id'],status=error));continue
  dna,prot,mapping=result
  ps=[p for p in by_gene[t['gene_id']] if p['full_sequence']==prot]
  if not ps:txaudit.append(dict(transcript_id=tid,gene_id=t['gene_id'],status='translated_full_protein_mismatch'));continue
  txaudit.append(dict(transcript_id=tid,gene_id=t['gene_id'],status='exact_reference_match',uniprots=';'.join(sorted(set(p['uniprot'] for p in ps)))))
  pos,ix=bychr[t['chrom']]
  # CDS phase is validated; do not drop phase bases from each exon.
  regions=[(a,b,cum,False) for a,b,cum in mapping]
  stop_segments=sorted(t['stop'],reverse=t['strand']=='-');stopcum=len(dna);stopdna=''
  for a,b in stop_segments:
   s=genome.fetch(t['chrom'],a-1,b).upper();s=rc(s) if t['strand']=='-' else s
   regions.append((a,b,stopcum,True));stopcum+=len(s);stopdna+=s
  for a,b,cum,isstop in regions:
   lo=np.searchsorted(pos,a);hi=np.searchsorted(pos,b,side='right')
   for vi in ix[lo:hi]:
    r=v.loc[vi];offset=cum+(r.pos1-a if t['strand']=='+' else b-r.pos1);alt=r.alt if t['strand']=='+' else rc(r.alt)
    coding=dna+stopdna;aa0=offset//3;codon0=aa0*3;refcodon=coding[codon0:codon0+3]
    if len(refcodon)!=3:continue
    expected=r.ref if t['strand']=='+' else rc(r.ref)
    if coding[offset]!=expected:raise ValueError('CDS/reference mismatch '+tid+':'+r.variant_id)
    altcodon=refcodon[:offset%3]+alt+refcodon[offset%3+1:]
    refaa=str(Seq(refcodon).translate());altaa=str(Seq(altcodon).translate())
    consequence='synonymous' if refaa==altaa else 'stop_loss' if refaa=='*' else 'stop_gain' if altaa=='*' else 'missense'
    for p in ps:
     inside=p['fragment_start0']<=aa0<p['fragment_end0'];status='ready' if inside and consequence in ('missense','stop_gain','stop_loss') else 'outside_selected_fragment' if not inside else 'excluded_consequence'
     wt,mut,crop0='','',0
     if status=='ready':
      fi=aa0-p['fragment_start0'];wt,crop0=centered_crop(p['sequence'],fi);mi=fi-crop0
      if consequence=='missense':mut=wt[:mi]+altaa+wt[mi+1:]
      elif consequence=='stop_gain':mut=wt[:mi]
      else:status='stop_loss_requires_extension_not_in_selected_fragment'
      if not mut:status='empty_mutant_fragment'
     rows.append(dict(variant_id=r.variant_id,protein_id=p['protein_id'],uniprot=p['uniprot'],tf=p['tf'],gene_id=t['gene_id'],transcript_id=tid,
       transcript_priority=t['priority'],strand=t['strand'],cds_pos1=offset+1,aa_pos1=aa0+1,ref_codon=refcodon,alt_codon=altcodon,
       ref_aa=refaa,alt_aa=altaa,consequence=consequence,in_selected_fragment=inside,status=status,
       fragment_start0=p['fragment_start0'],fragment_end0=p['fragment_end0'],crop_start0=crop0,wt_sequence=wt,alt_sequence=mut))
 allrows=pd.DataFrame(rows)
 allrows=allrows.merge(v,on='variant_id',validate='many_to_one')
 # Preserve ALL transcript consequences; main analysis deterministically takes the preferred matching transcript per SNP/fragment.
 allrows=allrows.sort_values(['variant_id','protein_id','transcript_priority','transcript_id'])
 selected=allrows.drop_duplicates(['variant_id','protein_id']).copy()
 selected['mapping_id']=[digest(a+'|'+b)[:24] for a,b in zip(selected.variant_id,selected.protein_id)]
 pd.DataFrame(txaudit).to_csv(RUN/'qc/transcript_validation.csv',index=False)
 parquet(allrows,RUN/'prepared/coding_all_transcripts.parquet');return selected

def main():
 (RUN/'prepared').mkdir(parents=True,exist_ok=True);(RUN/'qc').mkdir(exist_ok=True)
 genome=pysam.FastaFile(str(SHARE/'clinvar/hg38.fa'))
 v=read_variants(genome);print('SNVs',len(v),flush=True)
 p=read_proteins();genes,tx=gtf_data(set(p.gene_id));print('Proteins',len(p),'genes',len(genes),'transcripts',len(tx),flush=True)
 parquet(genes,RUN/'prepared/promoters.parquet')
 promoter,links=promoter_variants(v,genes,genome);print('Promoter',len(promoter),flush=True)
 coding=coding_variants(v,p,tx,genome)
 coding['also_promoter']=coding.variant_id.isin(promoter.variant_id)
 promoter['also_tf_coding']=promoter.variant_id.isin(coding.variant_id)
 promoter['also_selected_tf_fragment']=promoter.variant_id.isin(coding.loc[coding.in_selected_fragment,'variant_id'])
 parquet(coding,RUN/'prepared/coding_variants.parquet');parquet(promoter,RUN/'prepared/promoter_variants.parquet');parquet(links,RUN/'prepared/promoter_gene_links.parquet')
 motifs=load_motifs(p);save_json(motifs,RUN/'prepared/motifs.json')
 samples=[]
 for m in motifs:
  for i,s in enumerate(sample_dna(m)):samples.append(dict(source=m['source'],protein_id=m['protein_id'],tf=m['tf'],sample_index=i,dna_seq=s,motif_id=m['motif_id']))
 parquet(pd.DataFrame(samples),RUN/'prepared/sampled_dna.parquet')
 ready=coding[coding.status=='ready']
 interfaces=ready[['mapping_id','variant_id','protein_id','uniprot','tf','transcript_id','aa_pos1','ref_aa','alt_aa','consequence','label','wt_sequence','alt_sequence']].copy()
 full_map=p.set_index('protein_id').full_sequence.to_dict()
 interfaces['genome_build']='GRCh38'
 interfaces['full_wt_sequence']=interfaces.protein_id.map(full_map)
 interfaces['full_alt_sequence']=[wt[:pos-1]+alt+wt[pos:] if consequence=='missense' else wt[:pos-1] for wt,pos,alt,consequence in zip(interfaces.full_wt_sequence,interfaces.aa_pos1,interfaces.alt_aa,interfaces.consequence)]
 for method in ['esm2_fitness','alphamissense']:
  d=interfaces.copy();d['applicable']=d.consequence.eq('missense') if method=='alphamissense' else True;d['score']=np.nan;d['score_direction']='';d['implementation_version']='';d.to_csv(RUN/f'prepared/{method}_interface.csv',index=False)
 summary={'snvs':len(v),'promoter_snvs':len(promoter),'promoter_gene_links':len(links),'protein_coding_genes':len(genes),'usable_fragments':len(p),
  'coding_rows':len(coding),'coding_ready_rows':len(ready),'coding_ready_unique_snvs':ready.variant_id.nunique(),
  'coding_status':coding.status.value_counts().to_dict(),'consequences':coding.consequence.value_counts().to_dict(),
  'ready_consequences':ready.consequence.value_counts().to_dict(),'labels':v.label.value_counts().to_dict(),
  'promoter_labels':promoter.label.value_counts().to_dict(),'motifs':dict(collections.Counter(m['source'] for m in motifs)),
  'coordinate_convention':'variant_id/pos1 are VCF 1-based; all interval and crop fields ending 0 are 0-based half-open',
  'sampling':'100 independent PWM samples per fragment/source, motif centered in 40bp, file background flanks, SHA256-derived seed42'}
 save_json(summary,RUN/'qc/preparation_summary.json');print(summary,flush=True)
 from provenance import write_input_manifest
 write_input_manifest()
if __name__=='__main__':main()
