"""Numerical and scientific-contract checks; no external data required."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import ROOT, analyze, pwm_scores, regress, bh, variant_parts, is_snv


class Contracts(unittest.TestCase):
    def test_three_fragment_pipeline_averages_before_regression(self):
        (ROOT / '.cache/tmp').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache/tmp') as tmp:
            out = Path(tmp)
            (out / 'prepared').mkdir()
            for method in ['seq2kd', 'pwm']:
                (out / f'scores/{method}').mkdir(parents=True)
            pd.DataFrame({'variant_id': ['a', 'b', 'c'], 'group': ['aggregate']*3,
                          'outcome': [1., 2., 3.]}).to_csv(out / 'prepared/mpra_observations.csv', index=False)
            pd.DataFrame({'tf': ['TF']*3, 'protein_id': ['f1', 'f2', 'f3']}).to_csv(out / 'prepared/proteins.csv', index=False)
            pd.DataFrame({'tf': ['TF'], 'motif_id': ['m1']}).to_csv(out / 'prepared/pwm_manifest.csv', index=False)
            # Individual fragment effects disagree. Their mean is [1, 2, 3].
            for feature, effects in [('f1', [9., 0., -3.]), ('f2', [-6., 6., 3.]), ('f3', [0., 0., 9.]), ('m1', [1., 2., 3.])]:
                method = 'pwm' if feature == 'm1' else 'seq2kd'
                pd.DataFrame({'variant_id': ['a', 'b', 'c'], 'ref_score': [0.]*3,
                              'alt_score': effects, 'feature_id': [feature]*3}).to_parquet(out / f'scores/{method}/mpra.{feature}.parquet', index=False)
            analyze({'percentages': [100], 'binding_rule': 'combined', 'min_observations': 2}, out, ['mpra'])
            row = pd.read_csv(out / 'analysis/regressions.csv').query('method == "seq2kd"').iloc[0]
            self.assertEqual(row.n_features, 3)
            self.assertEqual(row.n, 3)
            self.assertAlmostEqual(row.slope, 1.)
            self.assertAlmostEqual(row.r_value, 1.)

    def test_pwm_matches_independent_exhaustive_strand_scan(self):
        rng = np.random.RandomState(3)
        pwm = rng.normal(size=(5, 4))
        seqs = ['ACGTGATCGATCGACGTAGC', 'TTTGGGACGTACGTTAGCCC']
        expected = []
        for s in seqs:
            rc = s.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
            expected.append(max(sum(pwm[j, 'ACGT'.index(strand[i+j])] for j in range(5))
                                for strand in [s, rc] for i in range(len(s)-4)))
        np.testing.assert_allclose(pwm_scores(seqs, pwm), expected)
        np.testing.assert_allclose(pwm_scores(seqs, pwm), pwm_scores([s.translate(str.maketrans('ACGT', 'TGCA'))[::-1] for s in seqs], pwm))

    def test_binding_threshold_and_allele_delta_sign(self):
        scores = pd.DataFrame({'variant_id': ['a', 'b', 'c', 'd'],
                               'ref_score': [1., 2., 3., 4.], 'alt_score': [11., 12., 13., 14.]})
        obs = pd.DataFrame({'variant_id': ['a', 'b', 'c', 'd'], 'group': ['aggregate']*4,
                            'outcome': [1., 2., 3., 4.]})
        combined = regress(scores, obs, 25, 'combined', 2)[0]
        legacy = regress(scores, obs, 25, 'legacy_ref', 2)[0]
        self.assertEqual(combined['threshold'], np.quantile(np.r_[scores.ref_score, scores.alt_score], .75))
        self.assertEqual(legacy['threshold'], 3.25)
        self.assertEqual(combined['n'], 2)
        # Constant allele difference must be reported, not crash or invent a regression.
        self.assertEqual(combined['status'], 'constant_predictor')
        scores.alt_score = [1., 4., 7., 10.]
        forward = regress(scores, obs, 100, 'combined', 2)[0]
        scores[['ref_score', 'alt_score']] = scores[['alt_score', 'ref_score']].to_numpy()
        reverse = regress(scores, obs, 100, 'combined', 2)[0]
        self.assertAlmostEqual(forward['r_value'], 1.)
        self.assertAlmostEqual(forward['r_value'], -reverse['r_value'])
        self.assertAlmostEqual(forward['p_value'], reverse['p_value'])

    def test_gene_specific_betas_are_retained(self):
        s = pd.DataFrame({'variant_id': ['a', 'b'], 'ref_score': [0., 0.], 'alt_score': [1., 2.]})
        o = pd.DataFrame({'variant_id': ['a', 'a', 'b'], 'group': ['aggregate']*3,
                          'outcome': [1., -1., 2.]})
        r = regress(s, o, 100, 'combined', 2)[0]
        self.assertEqual(r['n'], 3)
        self.assertEqual(r['n_snvs'], 2)
        self.assertAlmostEqual(r['slope'], 2.)

    def test_snv_gate_and_bh(self):
        z = variant_parts(pd.Series(['chr1_10_A_G_b38', 'chr1_11_AT_A_b38', 'chr2_12_N_T_b38']))
        self.assertEqual(is_snv(z).tolist(), [True, False, False])
        np.testing.assert_allclose(bh([.01, .04, .03, np.nan])[:3], [.03, .04, .04])


if __name__ == '__main__':
    unittest.main()
