"""Export four binding thresholds for the three requested methods; preserve PDFs."""
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
import plot_requested_median_pvalues as plot
from common import parquet, save_json
from motifs import encode, pwm_scores
from candidates import score_matrix

BASE = plot.RESULTS
RUN = BASE/'promoter_methods_all_thresholds'
PCTS = [1, 5, 10, 30]
V = REF = ALT = None


def scan(m):
    path = RUN/'candidates'/f"{m['tf']}.{m['motif_id']}.parquet"
    mat = np.log2((np.array(m['probabilities'])+1e-6)/np.array(m['background']))
    r, a = score_matrix(REF, mat), score_matrix(ALT, mat)
    assert np.allclose(r[:64], pwm_scores(REF[:64], m), atol=1e-10)
    assert np.allclose(a[:64], pwm_scores(ALT[:64], m), atol=1e-10)
    binding = np.maximum(r, a)
    cuts = {pct: float(np.percentile(binding, 100-pct)) for pct in PCTS}
    keep = binding >= cuts[30]
    d = V.loc[keep, ['variant_id']].copy()
    d['tf'] = m['tf']
    d['abs_delta'] = np.abs(a[keep]-r[keep])
    for pct in PCTS:
        d[f'top{pct}'] = binding[keep] >= cuts[pct]
    old = pd.read_parquet(BASE/'jaspar_promoter/candidates'/path.name)
    check = d.loc[d.top5].set_index('variant_id').sort_index()
    old = old.set_index('variant_id').sort_index()
    assert check.index.equals(old.index)
    np.testing.assert_allclose(check.abs_delta, old.abs_delta, atol=1e-12, rtol=0)
    parquet(d, path)
    return str(path), cuts


def main():
    global V, REF, ALT
    for sub in ['candidates', 'variant_summaries']:
        (RUN/sub).mkdir(parents=True, exist_ok=True)
    frozen = {str(p): plot.sha256(p) for p in plot.OUT.glob('*.pdf')}
    annpath = BASE/'prepared/promoter_variants.parquet'
    motifpath = BASE/'jaspar_coding/prepared/motifs.json'
    V = pd.read_parquet(annpath)
    assert not V.variant_id.duplicated().any()
    REF, ALT = encode(V.ref_seq), encode(V.alt_seq)
    motifs = json.loads(motifpath.read_text())
    unique = {(m['tf'], m['motif_id']): m for m in motifs}
    score_matrix(REF[:1], np.ones((1,4)))
    print(f'Scanning {len(unique)} PWM motifs', flush=True)
    records = []
    with mp.get_context('fork').Pool(16) as pool:
        for item in pool.imap_unordered(scan, list(unique.values())):
            records.append(item)
            if len(records) % 100 == 0:
                print(f'Validated {len(records)}/{len(unique)} PWM motifs', flush=True)
    save_json(records, RUN/'thresholds.json')
    pairs = pd.concat([pd.read_parquet(p) for p, _ in records], ignore_index=True)
    for pct in PCTS:
        tf = pairs.loc[pairs[f'top{pct}']].groupby(['variant_id','tf'], as_index=False).abs_delta.max()
        d = tf.groupby('variant_id', as_index=False).agg(mean=('abs_delta','mean'),n_tf=('tf','size'))
        d = d.merge(V[['variant_id','label']], validate='one_to_one')
        if pct == 5:
            old = pd.read_parquet(BASE/'analysis/variant_summaries/promoter.pwm_jaspar.top5.mean.parquet').set_index('variant_id').sort_index()
            new = d.set_index('variant_id').sort_index()
            assert old.index.equals(new.index)
            np.testing.assert_allclose(old['mean'], new['mean'], atol=1e-12, rtol=0)
        parquet(d, RUN/'variant_summaries'/f'promoter.pwm_jaspar.top{pct}.mean.parquet')
    rows = []
    methods = [('seq2kd_streme','Seq2Kd–STREME','Predicted binding affinity (pKd)',r'$|\Delta pK_d|$'),
               ('probound','ProBound','Relative binding affinity (Ka)',r'$|\log_{10}(K_{a,\mathrm{ALT}}/K_{a,\mathrm{REF}})|$'),
               ('pwm_jaspar','PWM–JASPAR','PWM motif score (log2-odds)',r'$|\Delta\mathrm{motif\ score}|$')]
    for method, name, quantity, formula in methods:
        for pct in PCTS:
            endpoint = f'promoter.{method}.top{pct}.mean'
            path = (RUN if method == 'pwm_jaspar' else BASE/'analysis')/'variant_summaries'/f'{endpoint}.parquet'
            d = pd.read_parquet(path)
            assert not d.variant_id.duplicated().any()
            d = d.loc[d.label.isin(['Benign','Pathogenic'])]
            assert np.isfinite(d['mean']).all()
            b = d.loc[d.label.eq('Benign'),'mean'].to_numpy(dtype=float)
            p = d.loc[d.label.eq('Pathogenic'),'mean'].to_numpy(dtype=float)
            y = d.label.eq('Pathogenic').astype(int)
            auc = float(roc_auc_score(y,d['mean']))
            test = mannwhitneyu(p,b,alternative='two-sided',method='auto')
            assert np.isclose(auc,test.statistic/(len(b)*len(p)),atol=1e-12,rtol=0)
            rows.append(dict(endpoint=endpoint,method=name,threshold_pct=pct,quantity=quantity,formula=formula,
                benign_n=len(b),pathogenic_n=len(p),benign_median=float(np.median(b)),
                pathogenic_median=float(np.median(p)),median_difference=float(np.median(p)-np.median(b)),
                auc_pathogenic_higher=auc,cliffs_delta_pathogenic_minus_benign=2*auc-1,
                auprc_average_precision=float(average_precision_score(y,d['mean'])),
                pathogenic_prevalence_pr_baseline=float(y.mean()),raw_p=float(test.pvalue),
                source=str(path),source_sha256=plot.sha256(path)))
    rows = pd.DataFrame(rows)
    reference = pd.read_csv(plot.OUT/'source_statistics.csv').set_index('endpoint')
    for r in rows.loc[rows.threshold_pct.eq(5)].itertuples():
        old = reference.loc[r.endpoint]
        assert (r.benign_n,r.pathogenic_n)==(old.benign_n,old.pathogenic_n)
        assert np.isclose(r.raw_p,old.raw_p,rtol=1e-8,atol=0)
        assert np.isclose(r.median_difference,old.median_difference,atol=1e-12,rtol=0)
    rows.to_csv(RUN/'method_threshold_statistics.csv',index=False)
    upper = plot.auc_upper(rows.auc_pathogenic_higher)
    assert rows.auc_pathogenic_higher.min() >= .5
    fig, axes = plot.plt.subplots(1,3,figsize=(17,7),sharey=True)
    colors = ['#376895','#BB8426','#63856B']
    for ax, (method,name,quantity,formula), color in zip(axes,methods,colors):
        view = rows.loc[rows.method.eq(name)]
        x = np.arange(4)
        bars = ax.bar(x,view.auc_pathogenic_higher-.5,bottom=.5,width=.58,color=color,zorder=3)
        ax.set_title(name+'\n'+quantity+'\n'+formula,fontsize=12,pad=16)
        ax.set_xticks(x,[f'top {r.threshold_pct}%\nP n={r.pathogenic_n:,}\nB n={r.benign_n:,}\n\n'+f'{r.median_difference:+.5f}' for r in view.itertuples()],fontsize=10)
        ax.set_ylim(.5,upper)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in view.auc_pathogenic_higher],padding=5,fontsize=10)
        ax.spines['right'].set_visible(False)
        plot.decorate(ax)
    axes[0].set_ylabel('AUC')
    fig.suptitle('Promoter variants: methods and binding thresholds (mean across TFs)',y=.97,fontsize=15)
    fig.text(.5,.045,'P: Pathogenic; B: Benign. Values below sample counts: Median(P) − Median(B) of the method-specific absolute effect.\nAUC positive class: Pathogenic; higher absolute effects predict Pathogenic; AUC axis starts at 0.5.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.055,right=.985,bottom=.29,top=.75,wspace=.10)
    output = plot.OUT/'03_promoter_methods_all_thresholds_auc.pdf'
    assert not output.exists(), 'Preserve existing outputs'
    fig.savefig(output)
    plot.plt.close(fig)
    assert frozen == {p:plot.sha256(plot.Path(p)) for p in frozen}
    save_json(dict(complete=True,original_pdfs_unchanged=True,pdf_fonttype=plot.mpl.rcParams['pdf.fonttype'],
        auc_limits=[.5,upper],top5_reproduced=True,scanner_crosscheck=True,
        inputs={str(p):plot.sha256(p) for p in [annpath,motifpath]},
        original_pdf_hashes=frozen,script_sha256=plot.sha256(plot.Path(__file__)),output_pdf=str(output)),RUN/'COMPLETE.json')
    print(rows[['method','threshold_pct','benign_n','pathogenic_n','auc_pathogenic_higher','median_difference']].to_string(index=False),flush=True)
    print('NEW PDF: '+str(output),flush=True)


if __name__ == '__main__':
    main()
