import unittest,tempfile
from pathlib import Path
import numpy as np,pandas as pd
import prepare
from prepare import transcript_cds,coding_variants,stars
from motifs import pwm_scores,encode,sample_dna
from candidates import score_matrix
from common import rc,normalize_chromosome
class FakeGenome:
 def __init__(self,mapping):self.mapping=mapping
 def fetch(self,c,a,b):return ''.join(self.mapping[i] for i in range(a+1,b+1))
class Tests(unittest.TestCase):
 def test_tf_stratification_exact_and_no_overlap(self):
  from tf_stratified import stratified_rank_sum
  d=pd.DataFrame(dict(variant_id=['a','b','c','d'],tf=['A','A','B','B'],value=[100.,101.,0.,1.],label=['Benign','Pathogenic','Benign','Pathogenic']))
  result=stratified_rank_sum(d,'Benign','Pathogenic','unit')
  self.assertEqual(result['n_tf'],2);self.assertAlmostEqual(result['p'],.5);self.assertAlmostEqual(result['effect'],-1.)
  d['label']=['Benign','Benign','Pathogenic','Pathogenic'];self.assertIsNone(stratified_rank_sum(d,'Benign','Pathogenic','unit'))

 def test_mitochondrial_aliases(self):
  for name in ["MT","M","chrMT","chrM"]:self.assertEqual(normalize_chromosome(name),"chrM")
  self.assertEqual(normalize_chromosome("1"),"chr1");self.assertEqual(normalize_chromosome("chrX"),"chrX")
 def test_negative_spliced_codon_and_phase(self):
  mapping={i:b for i,b in zip(range(101,105),'CCAT')}|{i:b for i,b in zip(range(81,86),'TTTAG')}
  g=FakeGenome(mapping);t=dict(transcript_id='tx',gene_id='g',chrom='chr1',strand='-',cds=[(81,85,2),(101,104,0)],stop=[],priority=0)
  data,error=transcript_cds(t,g);self.assertEqual(error,'');self.assertEqual(data[0],'ATGGCTAAA');self.assertEqual(data[1],'MAK')
  proteins=pd.DataFrame([dict(protein_id='p',uniprot='u',tf='TF',gene_id='g',full_sequence='MAK',sequence='MAK',fragment_start0=0,fragment_end0=3)])
  variants=pd.DataFrame([dict(variant_id='chr1:101:C>T',chrom='chr1',pos1=101,ref='C',alt='T',label='Pathogenic')])
  old=prepare.RUN
  with tempfile.TemporaryDirectory() as d:
   prepare.RUN=Path(d);(Path(d)/'qc').mkdir()
   try:r=coding_variants(variants,proteins,{'tx':t},g).iloc[0]
   finally:prepare.RUN=old
  self.assertEqual(r.wt_sequence,'MAK');self.assertEqual(r.alt_sequence,'MTK');self.assertEqual(r.cds_pos1,4);self.assertEqual(r.aa_pos1,2)
  t['cds']=[(81,85,0),(101,104,0)];self.assertIsNone(transcript_cds(t,g)[0])
 def test_stop_gain_truncation(self):
  g=FakeGenome({i:b for i,b in enumerate('ATGTGGAAA',1)})
  t=dict(transcript_id='tx',gene_id='g',chrom='chr1',strand='+',cds=[(1,9,0)],stop=[],priority=0)
  p=pd.DataFrame([dict(protein_id='p',uniprot='u',tf='TF',gene_id='g',full_sequence='MWK',sequence='MWK',fragment_start0=0,fragment_end0=3)])
  v=pd.DataFrame([dict(variant_id='v',chrom='chr1',pos1=5,ref='G',alt='A',label='Pathogenic')]);old=prepare.RUN
  with tempfile.TemporaryDirectory() as d:
   prepare.RUN=Path(d);(Path(d)/'qc').mkdir()
   try:r=coding_variants(v,p,{'tx':t},g).iloc[0]
   finally:prepare.RUN=old
  self.assertEqual(r.consequence,'stop_gain');self.assertEqual(r.alt_sequence,'M');self.assertEqual(r.status,'ready')
 def test_pwm_numba_and_reverse_complement(self):
  rng=np.random.default_rng(4);p=rng.dirichlet(np.ones(4),7);m=dict(probabilities=p.tolist(),background=[.2,.3,.3,.2])
  s=[''.join(x) for x in rng.choice(list('ACGT'),(15,40))];e=encode(s);mat=np.log2((p+1e-6)/np.array(m['background']))
  self.assertTrue(np.allclose(score_matrix(e,mat),pwm_scores(e,m),atol=1e-10));self.assertTrue(np.allclose(score_matrix(e,mat),score_matrix(encode([rc(x) for x in s]),mat)))
 def test_sampler_and_stars(self):
  m=dict(source='streme',protein_id='p',probabilities=[[1,0,0,0],[0,0,0,1]],background=[.25]*4)
  a=sample_dna(m);self.assertEqual(a,sample_dna(m));self.assertTrue(all(len(x)==40 and x[19:21]=='AT' for x in a));self.assertEqual(stars('criteria_provided,_multiple_submitters,_no_conflicts'),2)
if __name__=='__main__':unittest.main()
