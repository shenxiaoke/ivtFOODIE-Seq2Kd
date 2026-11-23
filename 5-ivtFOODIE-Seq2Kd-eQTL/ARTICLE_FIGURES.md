# Article figure entry points

The entries below illustrate how to trace a figure from workflow to source table.

| Figure example | Stage and source |
| --- | --- |
| `go_enrichment_all_p.png` | Joint GTEx/MPRA threshold-sensitivity run. `analysis/go_target_enrichment.csv` feeds the plot and its `.source.csv`. |
| `batch0__A__score_0.5__main__eQTL_MPRA_with_PWM.png` | `overexpression/association_set_enrichment.py` joins the joint regression table to the frozen batch 0 TF direction reference. The plot writes a `.source.tsv`. |
| `batch0_OE_vs_GO_area_venn.pdf` | `overexpression/plot_oe_vs_go.py` uses the TF classifications, eligibility and GOALL tables. It writes overlap counts and named memberships. |

The recommended GTEx-only entry point produces `go_enrichment_eqtl_top5` using the top 5% setting.
