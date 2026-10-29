#!/usr/bin/env python3
"""Read-only audit of QNTC INT4 .qblk priority groups and optional spatial dependencies.

Requires only Python stdlib for QUIC-stage report; NumPy and the user's util_gau
viewer module are needed only for optional Gaussian spatial overlap analysis.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import struct
import sys

HEADER = struct.Struct('!5sIBHH')
RECORD = struct.Struct('!If32s')
N_BLOCKS = 1556
HASH_BLOCKS = 30976
TOTAL_BLOCKS = 31120


def read_stage(p, ntc, priority):
    raw = p.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f'Short stage: {p}')
    magic, frame, part, expected, count = HEADER.unpack_from(raw)
    if (magic, frame, part, expected) != (b'QNP01', ntc, priority, N_BLOCKS):
        raise ValueError(f'Unexpected header: {p} : {(magic, frame, part, expected)}')
    if len(raw) != HEADER.size + count * RECORD.size:
        raise ValueError(f'Wrong file size: {p}')
    recs = [RECORD.unpack_from(raw, HEADER.size + i * RECORD.size)
            for i in range(count)]
    ids = [r[0] for r in recs]
    if len(ids) != len(set(ids)) or any(i >= TOTAL_BLOCKS for i in ids):
        raise ValueError(f'Invalid or repeated record IDs within {p}')
    return recs


def balanced_tiles(xyz, n):
    """Bit-for-bit spatial assignment rule in rank_ntc_frames.py."""
    import numpy as np
    tiles = [np.arange(len(xyz), dtype=np.int64)]
    while len(tiles) < n:
        _, i = max((len(ids), i) for i, ids in enumerate(tiles) if len(ids) > 1)
        ids = tiles.pop(i)
        axis = int(np.argmax(np.ptp(xyz[ids], axis=0)))
        ordered = ids[np.argsort(xyz[ids, axis], kind='stable')]
        mid = len(ordered) // 2
        tiles.extend([ordered[:mid], ordered[mid:]])
    tiles.sort(key=lambda ids: tuple(np.mean(xyz[ids], axis=0).tolist()))
    return tiles


def compute_spatial_membership(args, frame, rows):
    """Compute block->P0..P3 structural links, not numerical motion effects."""
    import numpy as np
    if frame != 114:
        raise ValueError('The provided animated structural map was validated for NTC114 only')
    if not args.map.is_file():
        raise FileNotFoundError(f'No structural map: {args.map}')
    schedule = args.orders / f'NTC_{frame:06d}.npz'
    if not schedule.is_file():
        raise FileNotFoundError(f'No tile priority schedule: {schedule}')
    if not args.init_ply.is_file():
        raise FileNotFoundError(f'No persistent Gaussian scene: {args.init_ply}')
    sys.path.insert(0, str(args.viewer))
    import util_gau
    xyz = np.asarray(util_gau.load_ply(str(args.init_ply)).xyz, dtype=np.float32)
    with np.load(schedule, allow_pickle=False) as s:
        tiers = np.asarray(s['tile_tiers'], dtype=np.uint8)
        if len(tiers) != 32 or (tiers > 3).any():
            raise ValueError('Invalid 32-tile schedule')
    tiles = balanced_tiles(xyz, 32)
    gaussian_tier = np.empty(len(xyz), dtype=np.uint8)
    for t, ids in enumerate(tiles):
        gaussian_tier[ids] = tiers[t]
    with np.load(args.map, allow_pickle=False) as m:
        blocks = m['block_ids'].astype(np.int64)
        offsets = m['offsets'].astype(np.int64)
        gaussians = m['gaussian_ids'].astype(np.int64)
        if len(offsets) != len(blocks) + 1 or len(xyz) != int(m['gaussian_count']):
            raise ValueError('Structural mapping / scene mismatch')
        if (gaussians < 0).any() or (gaussians >= len(xyz)).any():
            raise ValueError('Invalid Gaussian IDs in map')
        spatial = {}
        for i, bid in enumerate(blocks):
            lo, hi = int(offsets[i]), int(offsets[i+1])
            spatial[int(bid)] = np.bincount(gaussian_tier[gaussians[lo:hi]], minlength=4).tolist()
    membership_sets = [set() for _ in range(4)]
    for row in rows:
        bid = row['block_id']
        if bid >= HASH_BLOCKS:
            row.update({f'gaussians_in_P{x}': 'shared_MLP' for x in range(4)})
        else:
            c = spatial.get(bid)
            if c is None:
                raise ValueError(f'Block {bid} from QUIC absent in structural NTC114 map')
            row.update({f'gaussians_in_P{x}': int(c[x]) for x in range(4)})
            for t in range(4):
                if c[t]:
                    membership_sets[t].add(bid)
    return membership_sets, tiers


def main():
    h = Path.home()
    root = h / 'snap/firefox/common/QNTC_dynamic_299_priorities_v1'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frame', type=int, default=114, help='Inspect one NTC (default 114)')
    p.add_argument('--all', action='store_true', help='Read every NTC 0-298')
    p.add_argument('--stages', type=Path, default=root/'stages')
    p.add_argument('--orders', type=Path, default=root/'orders')
    p.add_argument('--out', type=Path, default=h/'snap/firefox/common/QNTC_priority_audit')
    p.add_argument('--print-ids', action='store_true', help='Print every original block ID to terminal')
    p.add_argument('--with-tiles', action='store_true', help='Also cross HashGrid blocks with NTC114 Gaussian tile colors')
    p.add_argument('--map', type=Path, default=h/'snap/firefox/common/QNTC_ntc114_animated_map_v2/ntc_114_hash_block_to_gaussian_ids.npz')
    p.add_argument('--init-ply', type=Path, default=h/'qntc_scenes/flame_steak_official/init_3dgs.ply')
    p.add_argument('--viewer', type=Path, default=h/'3DGS-QuaNTiC-Demo')
    a = p.parse_args()
    if a.with_tiles and a.all:
        p.error('--with-tiles requires one selected frame (NTC114); mapping not available for every NTC')
    if not 0 <= a.frame <= 298:
        p.error('--frame must be in 0..298')
    frames = range(299) if a.all else [a.frame]
    a.out.mkdir(parents=True, exist_ok=True)
    summary = []
    all_rows = []
    for f in frames:
        seen = set()
        counts = []
        rows = []
        concatenated = []
        for st in range(4):
            path = a.stages / f'NTC_{f:06d}_P{st}.qblk'
            recs = read_stage(path, f, st)
            counts.append(len(recs))
            ids = [r[0] for r in recs]
            if seen.intersection(ids):
                raise ValueError(f'NTC{f}: overlapping IDs across priorities')
            seen.update(ids)
            concatenated.extend(ids)
            print(f'NTC {f:03} P{st}: {len(ids)} records ' +
                  f'(HashGrid={sum(i < HASH_BLOCKS for i in ids)}, MLP={sum(i >= HASH_BLOCKS for i in ids)})')
            if a.print_ids:
                print(f'   IDs: {ids}')
            for offset, (bid, scale, _packed) in enumerate(recs):
                rows.append({'ntc':f, 'transmission_priority': f'P{st}',
                             'position_in_stage':offset, 'block_id':int(bid),
                             'network':'HashGrid' if bid < HASH_BLOCKS else 'MLP',
                             'quant_scale':float(scale)})
        if len(seen) != N_BLOCKS:
            raise ValueError(f'NTC {f}: {len(seen)} blocks total, expected {N_BLOCKS}')
        order_file = a.orders/f'NTC_{f:06d}.npz'
        if order_file.is_file():
            import numpy as np
            with np.load(order_file, allow_pickle=False) as m:
                expected = m['ordered_block_ids'].tolist()
            if concatenated != expected:
                raise ValueError(f'NTC{f}: stage order differs from .npz ranking')
        summary.append({'ntc':f, 'P0':counts[0], 'P1':counts[1], 'P2':counts[2],
                        'P3':counts[3], 'total':len(seen)})
        all_rows.extend(rows)
        if a.with_tiles:
            membership, tiers = compute_spatial_membership(a, f, rows)
            print('\nSpatial structural coverage (overlaps are expected):')
            print('Tiles:', {f'P{x}':[i for i,t in enumerate(tiers) if t == x] for x in range(4)})
            for color, blocks in enumerate(membership):
                print(f'  P{color}: {len(blocks)} distinct HashGrid blocks touching colored Gaussians')
            spatial_file = a.out/f'ntc_{f:06d}_spatial_block_ids.csv'
            with spatial_file.open('w', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(['spatial_priority','distinct_hashgrid_block_count','original_block_ids'])
                for color, blocks in enumerate(membership):
                    writer.writerow([f'P{color}',len(blocks),' '.join(map(str,sorted(blocks)))])
            print('[SAVED]',spatial_file)
    raw_file = a.out / ('all_299_stage_block_ids.csv' if a.all else f'ntc_{a.frame:06d}_stage_block_ids.csv')
    with raw_file.open('w', newline='') as f:
        keys = ['ntc','transmission_priority','position_in_stage','block_id','network','quant_scale']
        if a.with_tiles:
            keys += [f'gaussians_in_P{i}' for i in range(4)]
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    counts_file = a.out/('all_299_stage_counts.csv' if a.all else f'ntc_{a.frame:06d}_stage_counts.csv')
    with counts_file.open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['ntc','P0','P1','P2','P3','total'])
        writer.writeheader();writer.writerows(summary)
    print('[SAVED]',raw_file)
    print('[SAVED]',counts_file)
    print('[READ-ONLY] No source checkpoint or receiver files changed')


if __name__ == '__main__':
    main()
