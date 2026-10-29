#!/usr/bin/env python3
"""Read-only per-Gaussian structural HashGrid access map for sparse NTC 114.

Output is potential parameter ACCESS, not numerical displacement contribution.
The GPU probe validates the CPU indexer against the installed tiny-cuda-nn.
Original scene, NTCs, viewer, QUIC and wire checkpoints are never changed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np

N_PARAMS = 1_991_680
HASH_PARAMS = 1_982_464
BLOCK = 64
HASH_BLOCKS = HASH_PARAMS // BLOCK
DEFAULT_BASE = Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat')
PRIMES = {
    'coherentprime': (1, 2654435761, 805459861),
    'prime': (1958374283, 2654435761, 805459861),
    'reversedprime': (2165219737, 1434869437, 2097192037),
}


def next_mult8(x: int) -> int:
    return ((x + 7) // 8) * 8


def encoding_layout(cfg: dict):
    grid_type = cfg.get('type', 'Hash').lower()
    if grid_type != 'hash' and cfg.get('otype', '').lower() not in ('hashgrid',):
        raise ValueError(f'This tool handles HashGrid only, got type={grid_type!r}')
    if cfg.get('stochastic_interpolation', False) or cfg.get('fixed_point_pos', False):
        raise ValueError('Stochastic interpolation or fixed_point_pos requires a different indexer')
    interp = cfg.get('interpolation', 'Linear').lower()
    if interp not in ('linear', 'smoothstep', 'nearest'):
        raise ValueError(f'Unsupported interpolation {interp!r}')
    htype = cfg.get('hash', 'CoherentPrime').lower()
    if htype not in PRIMES:
        raise ValueError(f'Unsupported hash type {htype!r}; do not guess')
    fpl = int(cfg.get('n_features_per_level', 2))
    levels = int(cfg.get('n_levels', cfg.get('n_features', 16 * fpl) // fpl))
    if fpl not in (1, 2, 4, 8) or not (1 <= levels <= 32):
        raise ValueError('Unexpected HashGrid feature or level count')
    base = int(cfg.get('base_resolution', 16))
    scale_factor = float(cfg.get('per_level_scale', 2.0))
    max_entries = 1 << int(cfg.get('log2_hashmap_size', 19))
    offset = 0
    layout = []
    for level in range(levels):
        # tiny-cuda-nn: exp2f(level*log2(per_level_scale))*base_resolution - 1
        scale = np.float32(np.exp2(np.float32(level) * np.float32(np.log2(np.float32(scale_factor)))) * np.float32(base) - np.float32(1.0))
        resolution = int(math.ceil(float(scale))) + 1
        entries = min(next_mult8(resolution**3), max_entries)
        layout.append((level, offset, entries, resolution, scale))
        offset += entries
    if offset * fpl != HASH_PARAMS:
        raise ValueError(f'Encoding config produces {offset*fpl} parameters, expected {HASH_PARAMS}. Config/indexing mismatch; stopping.')
    return layout, fpl, interp, PRIMES[htype], htype


def accessed_for_positions(coords: np.ndarray, cfg: dict, layout, features, interpolation, primes):
    """Return (Gaussian offset within coords, absolute HashGrid 64-param block id)."""
    n = len(coords)
    all_g = []
    all_b = []
    n_corners = 1 if interpolation == 'nearest' else 8
    for _, level_offset, entries, resolution, level_scale in layout:
        # pos_grid=floorf(fmaf(scale, normalized_coordinate, 0.5f))
        scaled = (coords * np.float32(level_scale) + np.float32(0.5)).astype(np.float32)
        floored = np.floor(scaled)
        gp = floored.astype(np.int64)
        frac = scaled - floored
        for corner in range(n_corners):
            dx, dy, dz = corner & 1, (corner >> 1) & 1, (corner >> 2) & 1
            if interpolation == 'nearest':
                dx = dy = dz = 0
            if interpolation != 'nearest':
                # Corners with exactly zero interpolation weight are not accessed as a
                # meaningful numerical dependency (kernel still reads their values).
                weight = ((frac[:, 0] if dx else 1.0 - frac[:, 0]) *
                          (frac[:, 1] if dy else 1.0 - frac[:, 1]) *
                          (frac[:, 2] if dz else 1.0 - frac[:, 2]))
                valid = weight > 0.0
            else:
                valid = np.ones(n, dtype=bool)
            if not valid.any():
                continue
            g = np.nonzero(valid)[0].astype(np.uint32)
            x = gp[valid, 0].astype(np.uint64) + np.uint64(dx)
            y = gp[valid, 1].astype(np.uint64) + np.uint64(dy)
            z = gp[valid, 2].astype(np.uint64) + np.uint64(dz)
            # grid_index: use dense flattening until the table is smaller than full grid
            # (or resolution too big for uint32 stride), then use 32-bit xor hash.
            dense_stride = resolution**3
            if resolution <= 1625 and entries >= dense_stride:
                idx = (x + y * np.uint64(resolution) + z * np.uint64(resolution**2)) % np.uint64(entries)
            else:
                idx = ((x * np.uint64(primes[0])) ^ (y * np.uint64(primes[1])) ^
                       (z * np.uint64(primes[2]))) & np.uint64(0xffffffff)
                idx %= np.uint64(entries)
            b = ((np.uint64(level_offset) + idx) * np.uint64(features) // np.uint64(BLOCK)).astype(np.uint32)
            all_g.append(g)
            all_b.append(b)
    if not all_b:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)
    g = np.concatenate(all_g)
    b = np.concatenate(all_b)
    # Unique pairs: a block may be accessed repeatedly by different grid corners/levels.
    key = b.astype(np.uint64) * np.uint64(n) + g.astype(np.uint64)
    key = np.unique(key)
    return (key % np.uint64(n)).astype(np.uint32), (key // np.uint64(n)).astype(np.uint32)


def cuda_crosscheck(cfg, all_coords, in_bounds, layout, features, interp, primes, n_probes=12, seed=114):
    import torch
    import tinycudann as tcnn
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Do not label mapping exact without running verification.')
    enc = tcnn.Encoding(n_input_dims=3, encoding_config=cfg).to('cuda')
    enc.train()
    if enc.params.numel() != HASH_PARAMS:
        raise RuntimeError(f'Installed tiny-cuda-nn has {enc.params.numel()} parameters; expected {HASH_PARAMS}')
    good = np.nonzero(in_bounds)[0]
    if not len(good):
        raise RuntimeError('No Gaussian positions within NTC bounds')
    rng = np.random.default_rng(seed)
    ids = rng.choice(good, size=min(n_probes, len(good)), replace=False)
    checked = 0
    for gid in ids:
        xyz = all_coords[gid:gid + 1]
        _, expected = accessed_for_positions(xyz, cfg, layout, features, interp, primes)
        expected = set(map(int, expected))
        enc.zero_grad(set_to_none=True)
        # tcnn CUDA kernels often require a padded batch; repeating a single
        # point gives exactly the same support for that one Gaussian.
        x = torch.from_numpy(np.repeat(xyz, 128, axis=0)).to('cuda')
        y = enc(x).float()
        y.sum().backward()
        grad = enc.params.grad
        if grad is None:
            raise RuntimeError('tiny-cuda-nn returned no encoding gradients')
        detected = torch.nonzero((grad.detach().float().reshape(-1, BLOCK).abs() > 0).any(dim=1)).flatten().cpu().numpy()
        observed = set(map(int, detected))
        if observed != expected:
            a = sorted(expected - observed)
            b = sorted(observed - expected)
            raise RuntimeError(f'GPU INDEX CROSSCHECK FAILED at gaussian {gid}: '
                               f'expected-only={a[:15]}, gradient-only={b[:15]}. '
                               'Do not trust structural mapping until this is resolved.')
        checked += 1
        print(f'[CUDA CHECK] Gaussian {gid}: {len(expected)} structural blocks exactly match tiny-cuda-nn gradients', flush=True)
    print(f'[CUDA PASS] {checked}/{checked} individual-Gaussian encoding index checks match', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, default=DEFAULT_BASE / 'scenes/candidate')
    p.add_argument('--wire', type=Path, default=DEFAULT_BASE / 'wire')
    p.add_argument('--viewer', type=Path, default=Path.home() / '3DGS-QuaNTiC-Demo')
    p.add_argument('--ntc', type=int, default=114)
    p.add_argument('--out', type=Path, default=Path.home() / 'snap/firefox/common/QNTC_individual_gaussian_ntc114')
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--max-gaussians', type=int, default=0, help='0=all; nonzero for smaller trial only')
    p.add_argument('--cuda-probes', type=int, default=12)
    p.add_argument('--verify-only', action='store_true', help='Run CUDA checks without writing output')
    args = p.parse_args()
    if args.batch_size < 16 or args.cuda_probes < 1 or args.ntc < 0 or args.ntc > 298:
        p.error('Invalid batch size, probe count, or NTC index')
    scene = args.scene.expanduser().resolve()
    wire = args.wire.expanduser().resolve()
    config_path = scene / 'NTCs/config.json'
    ply_path = scene / 'init_3dgs.ply'
    checkpoint_path = wire / f'NTC_{args.ntc:06d}.pth'
    for f in (config_path, ply_path, checkpoint_path):
        if not f.is_file():
            raise FileNotFoundError(f)
    cfg = json.loads(config_path.read_text())['encoding']
    layout, features, interp, primes, htype = encoding_layout(cfg)
    import torch
    sys.path.insert(0, str(args.viewer.expanduser().resolve()))
    import util_gau
    positions = np.asarray(util_gau.load_ply(str(ply_path)).xyz, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Invalid initial Gaussian positions')
    wire_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    idx_key = 'model.params_sparse_int4_block_indices'
    if idx_key not in wire_state:
        raise ValueError(f'Expected key {idx_key}, got {list(wire_state)[:12]}')
    retained_all = np.asarray(wire_state[idx_key].detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    if len(retained_all) != 1556 or len(np.unique(retained_all)) != len(retained_all):
        raise ValueError(f'Expected 1556 distinct retained blocks, got {len(retained_all)}')
    retained_hash = np.sort(retained_all[retained_all < HASH_BLOCKS].astype(np.int32))
    mlp_ids = np.sort(retained_all[retained_all >= HASH_BLOCKS].astype(np.int32))
    lo = np.asarray(wire_state['xyz_bound_min'], dtype=np.float32).reshape(1, 3)
    hi = np.asarray(wire_state['xyz_bound_max'], dtype=np.float32).reshape(1, 3)
    if np.any(hi <= lo):
        raise ValueError('Invalid NTC bounds')
    if args.max_gaussians:
        positions = positions[:args.max_gaussians]
    coords = ((positions - lo) / (hi - lo)).astype(np.float32)
    in_bounds = np.logical_and(coords >= 0, coords <= 1).all(axis=1)
    print(f'[SCENE] NTC={args.ntc:03d}, Gaussians={len(positions)}, within bounds={in_bounds.sum()}, outside={len(positions)-in_bounds.sum()}', flush=True)
    print(f'[HASHGRID] levels={len(layout)} features/level={features} hash={htype} interp={interp} retained HashGrid={len(retained_hash)} MLP={len(mlp_ids)}', flush=True)
    cuda_crosscheck(cfg, coords, in_bounds, layout, features, interp, primes, args.cuda_probes)
    if args.verify_only:
        print('[VERIFIED ONLY] No output written')
        return
    retained_lookup = np.full(HASH_BLOCKS, -1, dtype=np.int32)
    retained_lookup[retained_hash] = np.arange(len(retained_hash), dtype=np.int32)
    all_keys = []
    ng = len(positions)
    for beg in range(0, ng, args.batch_size):
        end = min(beg + args.batch_size, ng)
        local = np.nonzero(in_bounds[beg:end])[0]
        if not len(local):
            continue
        g, b = accessed_for_positions(coords[beg:end][local], cfg, layout, features, interp, primes)
        b_index = retained_lookup[b]
        keep = b_index >= 0
        # Local g indexes into the filtered chunk; translate to global Gaussian IDs.
        gids = local[g[keep]].astype(np.uint64) + np.uint64(beg)
        # Unique pair across corner and overlapping levels (for same source Gaussian)
        keys = np.unique(b_index[keep].astype(np.uint64) * np.uint64(ng) + gids)
        all_keys.append(keys)
        if end == ng or end % (args.batch_size * 16) == 0:
            print(f'[SCAN] {end:,}/{ng:,} Gaussians', flush=True)
    if not all_keys:
        raise RuntimeError('No Gaussian-to-retained-block dependencies found')
    keys = np.concatenate(all_keys)
    keys.sort()
    # Each Gaussian belongs to a unique chunk, so no cross-chunk duplicates.
    row = (keys // np.uint64(ng)).astype(np.int32)
    gids = (keys % np.uint64(ng)).astype(np.int32)
    counts = np.bincount(row, minlength=len(retained_hash)).astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    if int(offsets[-1]) != len(gids):
        raise AssertionError('CSR count mismatch')
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'ntc_{args.ntc:03d}_hash_block_to_gaussian_ids.npz'
    np.savez_compressed(dest,
        block_ids=retained_hash, offsets=offsets, gaussian_ids=gids,
        gaussian_count=np.int32(ng), inside_bounds_ids=np.flatnonzero(in_bounds).astype(np.int32),
        mlp_retained_block_ids=mlp_ids,
        block_size=np.int32(BLOCK), ntc_index=np.int32(args.ntc))
    order = np.argsort(-counts, kind='stable')
    with (out_dir / f'ntc_{args.ntc:03d}_block_counts.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original_hashgrid_block_id', 'gaussian_count_structurally_accessing'])
        writer.writerows((int(retained_hash[i]), int(counts[i])) for i in order)
    report = {
        'ntc': args.ntc,
        'source_scene': str(scene), 'source_sparse_checkpoint': str(checkpoint_path),
        'gaussians': ng, 'gaussians_in_bounds': int(in_bounds.sum()),
        'gaussians_outside_bounds': int((~in_bounds).sum()),
        'retained_hashgrid_blocks': len(retained_hash),
        'retained_mlp_blocks_global_potential': len(mlp_ids),
        'structural_block_gaussian_pairs': int(offsets[-1]),
        'hashgrid_blocks_with_gaussians': int(np.count_nonzero(counts)),
        'mean_gaussians_per_retained_hash_block': float(counts.mean()),
        'median_gaussians_per_retained_hash_block': float(np.median(counts)),
        'most_accessed_blocks': [{'block_id': int(retained_hash[i]), 'gaussians': int(counts[i])} for i in order[:20]],
        'interpretation': 'Potential exact HashGrid lookup dependency at INITIAL Gaussian positions, crosschecked against installed tiny-cuda-nn. NOT movement delta or ablation sensitivity. Retained MLP blocks are global potential connections.',
        'animated_positions': False, 'not_included': 'Gaussian additions not among persistent init Gaussian IDs'
    }
    (out_dir / f'ntc_{args.ntc:03d}_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('[DONE] exact-lookup structural HashGrid pairs:', int(offsets[-1]), flush=True)
    print('[TOP10]', [(int(retained_hash[i]), int(counts[i])) for i in order[:10]], flush=True)
    print('[SAVED]', out_dir, flush=True)
    print('[UNCHANGED] original scene, NTCs, QUIC sender/receiver and viewer', flush=True)


if __name__ == '__main__':
    main()
