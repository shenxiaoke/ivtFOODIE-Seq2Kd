#!/usr/bin/env python3
"""FP32 paired inference, fixed pseudorandom crop per protein, resumable per TF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import types

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML

from pipeline import ROOT, output_path, sha256, save_json


def crop_start(protein_id, length, seed, width=512):
    # Stable across sharding, batching, resumes and Python hash randomization.
    value = int.from_bytes(hashlib.sha256(f'{seed}:{protein_id}'.encode()).digest()[:8], 'little')
    return random.Random(value).randrange(max(0, length - width) + 1)


def encode_dna(sequences):
    table = {'A': 2, 'T': 3, 'C': 4, 'G': 5}
    return torch.tensor([[7] + [table[b] for b in s] for s in sequences], dtype=torch.long)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--limit-proteins', type=int)
    p.add_argument('--datasets', nargs='+', choices=['eqtl', 'mpra'], default=['mpra'])
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    out = output_path(args.output)
    dest = output_path(out / 'scores/seq2kd')
    modeldir = Path(c['model_dir'])
    sys.path.insert(0, str(modeldir))
    from deepproteindna.Bert import DeepProteinDNA_BERT
    from deepproteindna.data_utils import ACIDS
    torch.set_num_threads(2)
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    random.seed(c['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Deterministic FP32 math, no autocast. CUBLAS workspace configured before CUDA use.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    cfg = YAML(typ='safe').load(modeldir / c['model_config'])
    model = DeepProteinDNA_BERT(**cfg['model'], embeddings_path=None)
    checkpoint = modeldir / c['checkpoint']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    incompat = model.load_state_dict(state, strict=False)
    # Fail closed on any architecture change instead of silently retaining random weights.
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f'Checkpoint/config architecture mismatch: {incompat}')
    model = model.cuda().eval()
    proteins = pd.read_csv(out / 'prepared/proteins.csv')
    if args.limit_proteins:
        proteins = proteins.iloc[:args.limit_proteins]
    proteins = proteins.iloc[args.shard::args.shards]
    with open(modeldir / c['embeddings'], 'rb') as f:
        embeddings = pickle.load(f)
    variants = {d: pd.read_csv(out / f'prepared/{d}_variants.csv') for d in args.datasets}
    dna = {d: encode_dna(pd.concat([v.ref_seq, v.alt_seq], ignore_index=True)) for d, v in variants.items()}
    signature = {'checkpoint_sha256': sha256(checkpoint), 'seed': c['seed'],
                 'crop_policy': 'one stable seeded random crop per protein shared across all alleles/SNVs/strands',
                 'source_model_sha256': sha256(modeldir / 'deepproteindna/Bert.py'),
                 'inference_sha256': sha256(__file__), 'precision': 'float32',
                 'batch_size': c['batch_size'],
                 'embeddings_path': str(modeldir / c['embeddings']),
                 'embeddings_size': (modeldir / c['embeddings']).stat().st_size,
                 'embeddings_mtime_ns': (modeldir / c['embeddings']).stat().st_mtime_ns,
                 'config_sha256': sha256(modeldir / c['model_config']),
                 'variants_sha256': {d: sha256(out / f'prepared/{d}_variants.csv') for d in variants}}
    save_json(signature, out / f'logs/inference_manifest_shard{args.shard}.json')
    starts, elapsed, scored = [], time.monotonic(), 0
    for protein in proteins.itertuples(index=False):
        name = protein.protein_id
        if name not in embeddings:
            raise KeyError(f'Missing embedding: {name}')
        emb = torch.as_tensor(embeddings[name], dtype=torch.float32)
        if emb.ndim != 3 or emb.shape[0] != 1 or emb.shape[2] != 1280:
            raise ValueError(f'Unexpected embedding shape: {name}, {emb.shape}')
        start = crop_start(name, emb.shape[1], c['seed'], model.use_protein_len)
        cropped = emb[:, start:start + model.use_protein_len, :].cuda()
        padded = torch.zeros((1, model.use_protein_len, 1280), device='cuda')
        padded[:, :cropped.shape[1]] = cropped
        # Encode once; cross attention remains unchanged and is still run per DNA sequence.
        with torch.inference_mode():
            fixed_encoded = model.transform_linear(padded) if model.transform_linear is not None else padded
            if model.segment_embedding:
                fixed_encoded = model.protein_segment_embedding(fixed_encoded, 1)

        def fixed_protein_encoder(self, protein_seq, protein_name=None):
            return fixed_encoded.expand(len(protein_seq), -1, -1)

        model._encode_protein = types.MethodType(fixed_protein_encoder, model)
        seq = ''.join(x for x in protein.sequence if x in ACIDS and x not in '0X')
        aa = [ACIDS.index(x) for x in seq[start:start + model.use_protein_len]]
        aa += [0] * (model.use_protein_len - len(aa))
        aa = torch.tensor([aa], device='cuda')
        starts.append({'protein_id': name, 'tf': protein.tf, 'embedding_length': emb.shape[1], 'start': start,
                       'end': min(start + model.use_protein_len, emb.shape[1])})
        for dataset, v in variants.items():
            target = dest / f'{dataset}.{name}.parquet'
            meta = target.with_suffix('.json')
            if target.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
                print('Cached', dataset, name, flush=True)
                continue
            values = []
            with torch.inference_mode():
                for offset in range(0, len(dna[dataset]), c['batch_size']):
                    batch = dna[dataset][offset:offset + c['batch_size']].cuda()
                    prediction = model(batch, aa.expand(len(batch), -1), None, None, [name] * len(batch))[1]
                    values.append(prediction.cpu().numpy().reshape(-1))
            values = np.concatenate(values)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite predictions: {name}/{dataset}')
            d = v[['variant_id']].copy()
            d['ref_score'], d['alt_score'] = values[:len(v)], values[len(v):]
            d['tf'], d['feature_id'] = protein.tf, name
            tmp = target.with_suffix('.partial.parquet')
            d.to_parquet(tmp, index=False)
            tmp.replace(target)
            save_json(signature, meta)
            scored += 2 * len(v)
            print(dataset, name, len(v), f'elapsed={time.monotonic()-elapsed:.1f}s', f'pairs_per_sec={scored/(time.monotonic()-elapsed):.1f}', flush=True)
        save_json(starts, out / f'logs/protein_crops_shard{args.shard}.json')
    print('Finished shard', args.shard, flush=True)


if __name__ == '__main__':
    main()
