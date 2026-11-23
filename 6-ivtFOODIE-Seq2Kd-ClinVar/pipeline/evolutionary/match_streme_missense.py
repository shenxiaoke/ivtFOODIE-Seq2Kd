#!/usr/bin/env python3
"""Make the common STREME coding cohort from validated Seq2Kd, ESM-2 and AlphaMissense output."""
import argparse
from pathlib import Path

import pandas as pd


def match(source, destination):
    data = pd.read_parquet(source)
    data = data[data.source.eq('streme') & data.consequence.eq('missense')]
    dbd = data[data.context.eq('dbd')].rename(columns={'esm_llr': 'dbd_LLR'})
    full = data.loc[data.context.eq('full_or_1022_window'), ['mapping_id', 'esm_llr']]
    joined = dbd.merge(full.rename(columns={'esm_llr': 'full_LLR'}), on='mapping_id', validate='one_to_one')
    joined = joined.dropna(subset=['dbd_LLR', 'full_LLR', 'am_score_validated', 'mean_abs_delta'])
    metadata = joined.groupby('variant_id')[['tf', 'label', 'review_stars']].nunique()
    joined = joined[joined.variant_id.isin(metadata.index[metadata.tf.eq(1)])]
    if (joined.groupby('variant_id')[['label', 'review_stars']].nunique() > 1).any().any():
        raise ValueError('Inconsistent clinical labels or review stars')
    matched = joined.groupby(['source', 'variant_id'], as_index=False).agg(
        dbd_LLR=('dbd_LLR', 'mean'), full_LLR=('full_LLR', 'mean'),
        am_score_validated=('am_score_validated', 'mean'), mean_abs_delta=('mean_abs_delta', 'mean'),
        label=('label', 'first'), tf=('tf', 'first'), review_stars=('review_stars', 'first'),
    )
    matched['negative_dbd_LLR'] = -matched.dbd_LLR
    destination.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(destination, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('association_input', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    match(args.association_input, args.destination)
