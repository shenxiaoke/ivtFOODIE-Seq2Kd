#!/usr/bin/env python3
"""Create AUC versions, with AUC axes starting at .5 and adaptive upper limits."""
import json
import plot_requested_median_pvalues as plot


def main():
    original_pdfs = {p.name: plot.sha256(p) for p in plot.OUT.glob('*.pdf')
                     if p.name in ['01_coding_motif_comparison.pdf',
                                   '02_promoter_motif_threshold_comparison.pdf',
                                   '03_promoter_method_pvalues.pdf']}
    sources = plot.pd.read_csv(plot.OUT/'source_statistics.csv').set_index('endpoint')
    tables = [plot.pd.read_csv(plot.OUT/name) for name in [
        '01_coding_auc_cliffs_delta.csv',
        '02_promoter_motif_threshold_auc_cliffs_delta.csv',
        '03_promoter_methods_auc_cliffs_delta.csv']]
    views = []
    for table in tables:
        d = sources.loc[table.endpoint].reset_index()
        d['auc_pathogenic_higher'] = table.auc_pathogenic_higher.to_numpy()
        assert (d.benign_n.to_numpy() == table.benign_n.to_numpy()).all()
        assert (d.pathogenic_n.to_numpy() == table.pathogenic_n.to_numpy()).all()
        assert (d.auc_pathogenic_higher > .5).all()
        views.append(d)
    plot.paired_bars(views[0], 'Coding variants: DNA motif sources (100 DNA)',
                     '01_coding_motif_comparison_auc.pdf', 7.8, use_auc=True)
    plot.paired_bars(views[1], 'Promoter variants: motif sources and binding thresholds (mean)',
                     '02_promoter_motif_threshold_comparison_auc.pdf', 12.5, use_auc=True)
    plot.methods_bars(views[2], use_auc=True)
    assert original_pdfs == {name: plot.sha256(plot.OUT/name) for name in original_pdfs}
    audit = dict(pdf_fonttype=plot.mpl.rcParams['pdf.fonttype'],
                 auc_positive_class='Pathogenic', original_pdfs_unchanged=True,
                 auc_limits=[dict(lower=.5,upper=plot.auc_upper(d.auc_pathogenic_higher)) for d in views],
                 auc_bar_baseline=.5,
                 script_sha256=plot.sha256(plot.Path(__file__)),
                 plotting_script_sha256=plot.sha256(plot.Path(plot.__file__)))
    (plot.OUT/'auc_figure_validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    main()
