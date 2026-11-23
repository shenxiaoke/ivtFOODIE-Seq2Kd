"""Score JASPAR promoter candidates and summarize top-5% PWM effects."""
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
from common import ROOT, filehash, save_json, parquet
from motifs import encode, pwm_scores
from candidates import score_matrix

BRANCH = ROOT/'results/jaspar_promoter'
V = REF = ALT = None


def scan(m):
    path = BRANCH/'candidates'/f"{m['tf']}.{m['motif_id']}.parquet"
    mat = np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
    ref = score_matrix(REF, mat)
    alt = score_matrix(ALT, mat)
    # Check optimized scanning against the original independent vectorized scanner.
    assert np.allclose(ref[:64], pwm_scores(REF[:64], m), atol=1e-10)
    assert np.allclose(alt[:64], pwm_scores(ALT[:64], m), atol=1e-10)
    binding = np.maximum(ref, alt)
    cutoff = float(np.percentile(binding, 95))
    keep = binding >= cutoff
    d = V.loc[keep, ['variant_id']].copy()
    d['tf'] = m['tf']; d['motif_id'] = m['motif_id']
    d['ref_score'] = ref[keep]; d['alt_score'] = alt[keep]
    d['delta'] = d.alt_score-d.ref_score; d['abs_delta'] = d.delta.abs()
    parquet(d, path)
    return dict(tf=m['tf'], motif_id=m['motif_id'], cutoff=cutoff, n=int(keep.sum()), path=str(path))


def main():
    global V, REF, ALT
    import analyze as original
    out = ROOT/'results/analysis'
    annpath = ROOT/'results/prepared/promoter_variants.parquet'
    motifpath = ROOT/'results/jaspar_coding/prepared/motifs.json'
    frozen = [out/'tables/statistical_tests.csv']
    before = {str(p):filehash(p) for p in frozen}
    V = pd.read_parquet(annpath)
    assert not V.variant_id.duplicated().any()
    REF = encode(V.ref_seq); ALT = encode(V.alt_seq)
    motifs = json.loads(motifpath.read_text())
    # Identical TF/motif assignments across protein fragments have identical scores.
    unique = {(m['tf'],m['motif_id']):m for m in motifs}
    jobs = [unique[k] for k in sorted(unique)]
    BRANCH.mkdir(parents=True, exist_ok=True)
    save_json(dict(universe=len(V),motif_features=len(jobs),fragment_assignments=len(motifs),
        input_hashes={str(p):filehash(p) for p in [annpath,motifpath]},
        definition='double-strand maximum PWM log2-odds; pseudocount 1e-6; top5% of max(REF,ALT) over all promoter SNVs, ties included; max absolute delta within TF; mean across selected TFs per unique SNV.'),BRANCH/'inputs.json')
    print(f'Scanning {len(jobs)} TF/motif features over {len(V)} SNVs',flush=True)
    score_matrix(REF[:1],np.ones((1,4)))
    with mp.get_context('fork').Pool(16) as pool:
        records=[]
        for r in pool.imap_unordered(scan,jobs):
            records.append(r)
            if len(records)%25==0: print(f'{len(records)}/{len(jobs)} motifs complete',flush=True)
    save_json(records,BRANCH/'thresholds.json')
    pairs=pd.concat([pd.read_parquet(r['path']) for r in records],ignore_index=True)
    assert not pairs.duplicated(['variant_id','tf','motif_id']).any()
    tf=pairs.groupby(['variant_id','tf'],as_index=False).agg(value=('abs_delta','max'))
    d=tf.groupby('variant_id',as_index=False).agg(mean=('value','mean'),n_tf=('tf','size'))
    d=d.merge(V[['variant_id','label','review_stars']],validate='one_to_one')
    assert np.isfinite(d['mean']).all()
    endpoint='promoter.pwm_jaspar.top5.mean'
    original.OUT=BRANCH/'analysis'; original.FIG=out/'figures'
    original.compare(d,endpoint,'mean','|delta PWM log2-odds score|',
        'Promoter / pwm_jaspar / top 5% / mean across TFs',primary=False)
    pd.DataFrame(original.TESTS).to_csv(BRANCH/'statistical_tests_unadjusted.csv',index=False)
    pd.DataFrame(original.SUMMARIES).to_csv(BRANCH/'group_distributions.csv',index=False)
    # Save chart data alongside the existing distributions for discoverability.
    parquet(d,out/'variant_summaries'/f'{endpoint}.parquet')
    old=pd.read_csv(out/'tables/statistical_tests.csv')
    selected=old[old.endpoint.isin([f'promoter.{m}.top5.mean' for m in ['seq2kd_streme','probound']]) & old.class_a.eq('Benign') & old.class_b.eq('Pathogenic')]
    new=pd.DataFrame(original.TESTS)
    new=new[new.class_a.eq('Benign') & new.class_b.eq('Pathogenic')]
    four=pd.concat([selected,new],ignore_index=True)[['endpoint','test','class_a','class_b','n_a','n_b','p','effect']]
    four.to_csv(out/'tables/promoter_top5_mean_methods_pathogenic_vs_benign.csv',index=False)
    assert before=={str(p):filehash(p) for p in frozen}
    from PIL import Image
    with Image.open(out/'figures'/f'{endpoint}.png') as im: im.verify()
    save_json(dict(complete=True,unique_snvs=len(d),tf_count=tf.tf.nunique(),
        group_counts=d.label.value_counts().to_dict(),original_files_unchanged=before,
        scanner_crosscheck='First 64 REF and ALT sequences for every motif; vectorized PWM vs JIT; atol 1e-10',
        script_sha256=filehash(__file__)),BRANCH/'COMPLETE.json')
    print(four.to_string(index=False),flush=True)


if __name__=='__main__': main()
