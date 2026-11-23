# TF overexpression direction assessment

The paper uses the TF Atlas perturbation expression data to assess whether the resulting direction assignments are suitable as external activator/repressor labels. Batch 0 direction labels depend on the expression reference, gene set and threshold. The outcomes are presented as an exploratory comparison.

1. Use the batch 0 expression analysis with its versioned TF target inventory, HGNC alias mapping and eligibility table. The TF reference audit contains both eligible TFs and the assigned direction at each score threshold.
2. For the association enrichment, run `association_set_enrichment.py --regressions … --proteins … --hgnc … --reference … --targets … --output …`. Supply the joint GTEx/MPRA four-threshold `regressions.csv`, its own prepared `proteins.csv`, the frozen HGNC table, the TF Atlas `reference_TF_audit.tsv`, and `target_universe.tsv`. The output includes the batch 0 A |S| ≥ 0.5 figure, membership tables and Fisher test checks.
3. For the overexpression/GO comparison, run `plot_oe_vs_go.py --classifications … --go … --eligibility … --targets … --output …`. Supply the overexpression `TF_classifications.tsv`, GOALL export from the eQTL workflow, batch 0 `ORF_eligibility_initial.tsv`, and the same target universe. The area plots show all eligible expression genes and ChIP-Atlas targets at two absolute log2 fold-change thresholds. The figure source records counts and member identities.

The overexpression reference tables come from the earlier TF Atlas expression processing. Keep the data versions and input checksums together when transferring them to a new server. These stages assess directional overlap for the observed reference and the paper does not use the resulting labels as ground truth.
