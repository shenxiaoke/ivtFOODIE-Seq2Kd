import unittest
import numpy as np,pandas as pd
from shared import *
from alphamissense import translate_tx
from assemble import seq_metrics
from association import centered_ranks,association
class Genome:
 def __init__(self,s):self.s=s
 def fetch(self,c,a,b):return self.s[a:b]
class Checks(unittest.TestCase):
 def test_minus_multiexon(self):
  # Reverse complement of ATG AAA TTC = GAA TTT CAT, interrupted by intron.
  genome=Genome('GAAT'+'GGGG'+'TTCAT')
  tx=dict(chrom='chr1',strand='-',cds=[(1,4,1),(9,13,0)])
  value,status=translate_tx(tx,genome);self.assertEqual(status,'ok');self.assertEqual(value[1],'MKF')
 def test_phase_rejects(self):
  t=dict(chrom='chr1',strand='+',cds=[(1,3,1)])
  self.assertEqual(translate_tx(t,Genome('ATG'))[1],'phase_inconsistent_or_partial')
 def test_dna_shift_versus_specificity(self):
  a=np.arange(100,dtype=float)/10
  shift=seq_metrics(a,a+.7);self.assertAlmostEqual(shift['mean_abs_delta'],.7);self.assertAlmostEqual(shift['sd_delta'],0);self.assertAlmostEqual(shift['dna_rank_loss'],0)
  specific=seq_metrics(a,a+np.sin(a));self.assertGreater(specific['sd_delta'],.5)
 def test_partial_correlation_matches_residual_regression(self):
  from association import partial_from_cov
  rng=np.random.default_rng(7);z=rng.normal(size=500);x=2*z+rng.normal(size=500);y=-3*z+.2*x+rng.normal(size=500)
  a=np.column_stack([x-x.mean(),y-y.mean(),z-z.mean()]);cov=a.T@a
  residuals=a[:,:2]-a[:,2:]@np.linalg.lstsq(a[:,2:],a[:,:2],rcond=None)[0]
  independent=np.corrcoef(residuals.T)[0,1]
  self.assertAlmostEqual(partial_from_cov(cov),independent,places=12)
 def test_robust_preference_reversal(self):
  a=np.arange(100,dtype=float)/10
  self.assertEqual(seq_metrics(a,a+.7)['robust_reversal_fraction'],0)
  self.assertEqual(seq_metrics(a,-a)['robust_reversal_fraction'],1)
 def test_tf_composition_control(self):
  d=pd.DataFrame(dict(tf=np.repeat(['a','b','c'],8),variant_id=[str(i) for i in range(24)],x=np.tile(np.arange(8),3)+np.repeat([0,100,200],8),y=np.tile(np.arange(8)[::-1],3)+np.repeat([0,100,200],8)))
  z=centered_ranks(d,['x','y']);self.assertAlmostEqual(np.corrcoef(z.T)[0,1],-1)
  result=association(d,'x','y',permutations=False);self.assertAlmostEqual(result['rho'],-1);self.assertEqual(result['n_variants'],24)
if __name__=='__main__':unittest.main()
