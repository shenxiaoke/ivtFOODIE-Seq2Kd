#!/usr/bin/env python3
"""Generate a self-contained HTML index linking all HT-SELEX deliverables."""

from __future__ import annotations

import argparse
import gzip
import heapq
import html
import json
import os
from pathlib import Path

import yaml


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def top_enriched(path: Path, column: str, n: int = 20):
    heap = []
    with open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx = header.index(column)
        count_idx = header.index(column.split("_log2_")[0] + "_count")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            score = float(fields[idx])
            item = (score, fields[0], int(fields[count_idx]))
            if len(heap) < n:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)


def link(target: str, label: str):
    return f'<a href="{html.escape(target)}">{html.escape(label)}</a>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    results = root / cfg.get("results_dir", "results")

    sections = []
    for tf, sample in cfg["samples"].items():
        rounds = [sample["baseline"]["name"]] + list(sample["rounds"])
        count_summary = results / "counts" / tf / "sequence_counts.summary.json"
        with open(count_summary) as handle:
            counts = json.load(handle)
        extraction_rows = []
        for round_name in rounds:
            stats_path = results / "sequences" / tf / f"{round_name}.extraction.json"
            with open(stats_path) as handle:
                stats = json.load(handle)
            total = stats.get("total_pairs", stats.get("total", 0))
            retained = stats.get("retained", 0)
            pct = 100 * retained / total if total else 0
            extraction_rows.append(
                f"<tr><td>{round_name}</td><td>{total:,}</td><td>{retained:,}</td>"
                f"<td>{pct:.2f}%</td><td>{counts['unique_sequences_per_round'][round_name]:,}</td></tr>"
            )
        final_round = cfg["motif"]["final_round"]
        enrich = results / "enrichment" / tf / "sequence_enrichment.tsv.gz"
        enrich_col = f"{final_round}_log2_enrichment_vs_cycle0_normalized"
        tops = top_enriched(enrich, enrich_col)
        top_rows = "".join(
            f"<tr><td class='seq'>{seq}</td><td>{count:,}</td><td>{score:.3f}</td></tr>"
            for score, seq, count in tops
        )
        pairwise_rows = []
        for previous, current in zip(rounds[:-1], rounds[1:]):
            comparison = f"{current}_vs_{previous}"
            pairwise_rows.append(
                f"<div><h4>{current} vs {previous}</h4>"
                f"<img src='../motifs/{tf}/streme_pairwise/{comparison}/motif1.logo.png' "
                f"alt='{tf} {comparison} STREME motif 1'>"
                f"<p>{link(f'../motifs/{tf}/streme_pairwise/{comparison}/streme.html', 'full STREME report')}</p></div>"
            )
        prevalence_path = results / "motifs" / tf / "hts_ibis" / "prevalence_by_cycle.tsv"
        prevalence_block = ""
        if prevalence_path.exists():
            prevalence_block = (
                "<h3>HTS-IBIS PWM prevalence by cycle</h3>"
                f"<img src='../motifs/{tf}/hts_ibis/prevalence_by_cycle.png' "
                f"alt='{tf} HTS-IBIS PWM prevalence by cycle'>"
                f"<p>{link(f'../motifs/{tf}/hts_ibis/prevalence_by_cycle.tsv', 'prevalence table')}</p>"
            )
        section = f"""
        <section><h2>{html.escape(tf)}</h2>
        <h3>Read processing and sequence diversity</h3>
        <table><thead><tr><th>Round</th><th>Input pairs/rows</th><th>Retained</th><th>Retention</th><th>Unique</th></tr></thead>
        <tbody>{''.join(extraction_rows)}</tbody></table>
        <p>Union unique sequences: {counts['union_unique_sequences']:,}.
        {link(f'../counts/{tf}/sequence_counts.tsv.gz', 'sequence counts')} ·
        {link(f'../enrichment/{tf}/sequence_enrichment.tsv.gz', 'Seq2Kd enrichment table')}</p>
        <h3>Motifs</h3>
        <div class='motifs'><div><h4>HTS-IBIS PWM</h4><img src='../motifs/{tf}/hts_ibis/logo.png' alt='{tf} HTS-IBIS logo'>
        <p>{link(f'../motifs/{tf}/hts_ibis/{tf}.pwm', 'PWM')} · {link(f'../motifs/{tf}/hts_ibis/{tf}.meme', 'MEME format')}</p></div>
        <div><h4>STREME</h4><p>{link(f'../motifs/{tf}/streme/streme.html', 'interactive STREME report')}</p>
        <p>{link(f'../motifs/{tf}/tomtom/tomtom.html', 'HTS-IBIS vs STREME comparison')}</p></div></div>
        <h3>Adjacent-round STREME motifs</h3>
        <div class='pairwise'>{''.join(pairwise_rows)}</div>
        {prevalence_block}
        <h3>Top {final_round} normalized enrichments vs cycle0</h3>
        <table><thead><tr><th>Sequence</th><th>{final_round} count</th><th>log2 enrichment</th></tr></thead><tbody>{top_rows}</tbody></table>
        </section>"""
        sections.append(section)

    versions = "unavailable"
    version_file = results / "report" / "software_versions.txt"
    if version_file.exists():
        versions = html.escape(version_file.read_text())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(cfg['project']['name'])} HT-SELEX report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}}h1,h2{{color:#173f5f}}
section{{border-top:3px solid #2a7f9e;margin-top:2.5rem;padding-top:1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #d8e0e8;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}.seq{{font-family:monospace}}
.motifs{{display:grid;grid-template-columns:1fr 1fr;gap:2rem}}.pairwise{{display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem}}
img{{max-width:100%;background:white}}pre{{background:#f4f6f8;padding:1rem;overflow:auto}}
.note{{background:#eef7fa;border-left:4px solid #2a7f9e;padding:1rem}}@media(max-width:750px){{.motifs{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(cfg['project']['name'])}</h1><p>{html.escape(cfg['project']['description'])}</p>
<p class="note">cycle0 is the shared unselected pool. Selected reads are quality-aware R1/R2 consensuses of the observed 40-nt insert. Both raw-count and library-size-normalized enrichments are reported.</p>
{''.join(sections)}
<section><h2>Software versions</h2><pre>{versions}</pre></section>
</body></html>"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    reference_mtime_ns = max(
        path.stat().st_mtime_ns
        for path in results.rglob("*")
        if path.is_file() and path != output
    )
    output_stat = output.stat()
    os.utime(output, ns=(output_stat.st_atime_ns, reference_mtime_ns))


if __name__ == "__main__":
    main()
