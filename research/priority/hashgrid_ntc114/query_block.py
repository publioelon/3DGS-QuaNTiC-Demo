#!/usr/bin/env python3
"""Show Gaussian IDs for a given original NTC HashGrid block ID."""
import argparse
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('block_id', type=int)
p.add_argument('--map', type=Path, default=Path.home()/'snap/firefox/common/QNTC_individual_gaussian_ntc114/ntc_114_hash_block_to_gaussian_ids.npz')
p.add_argument('--limit', type=int, default=30)
a = p.parse_args()
with np.load(a.map) as d:
    blocks, offsets, gids = d['block_ids'], d['offsets'], d['gaussian_ids']
    matches=np.flatnonzero(blocks == a.block_id)
    if len(matches) != 1:
        raise SystemExit(f'Block {a.block_id} not present among retained HashGrid blocks')
    i = int(matches[0]); selected = gids[int(offsets[i]):int(offsets[i+1])]
    print('Block:', a.block_id, 'Potentially affected persistent Gaussians:', len(selected))
    print('First Gaussian IDs:', selected[:a.limit].tolist())
