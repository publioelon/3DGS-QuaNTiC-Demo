#!/usr/bin/env python3
"""Build a compact, playback-time priority timeline from SAVED 3D SalFoM predictions.

Read-only input. This is viewer visualization data, not a QUIC block scheduler.
Uses exactly the 32 persistent partitions from the established NTC audit.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

DEFAULT_SALIENCY = Path('/data/3dgs_storage/QNTC_Saliency_Pilot/float3d_calibrated12_adjoint/raw_float32')
DEFAULT_SCENE = Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat/scenes/candidate')
DEFAULT_OUT = Path.home()/'snap/firefox/common/NTC_dynamic_saliency_timeline.json'


def spatial_tiles(xyz: np.ndarray, count: int):
    """Identical stable split/tie-breaking as NTC 114 priority analysis."""
    tiles = [np.arange(len(xyz), dtype=np.int64)]
    while len(tiles) < count:
        choices = [(len(ids), i) for i, ids in enumerate(tiles) if len(ids) > 1]
        if not choices:
            break
        _, idx = max(choices)
        ids = tiles.pop(idx)
        axis = int(np.argmax(np.ptp(xyz[ids], axis=0)))
        ordered = ids[np.argsort(xyz[ids, axis], kind='stable')]
        half = len(ordered)//2
        tiles.extend((ordered[:half], ordered[half:]))
    tiles.sort(key=lambda ids: tuple(np.mean(xyz[ids], axis=0).tolist()))
    return tiles


def compute_scores(values: np.ndarray, groups: list[np.ndarray]):
    """Average highest 1% of scores per tile, capturing sparse salient clusters.

    Compared with 99th percentile, this is less prone to all-zero ties when
    SalFoM saliency is sparse. All Gaussian records remain unchanged.
    """
    result = []
    for ids in groups:
        v = values[ids]
        k = max(1, int(math.ceil(len(v) * .01)))
        # kth largest including zeros, with zero baseline preserved.
        cutoff = len(v) - k
        hi = np.partition(v, cutoff)[cutoff:]
        result.append(float(hi.mean()))
    scores = np.asarray(result, dtype=np.float64)
    mx = float(scores.max())
    if mx <= 0:
        return None
    return (scores / mx).tolist()


def build(args):
    root = args.viewer.expanduser().resolve()
    scene = args.scene.expanduser().resolve()
    source = args.saliency_dir.expanduser().resolve()
    output = args.out.expanduser()
    ply = scene/'init_3dgs.ply'
    if not ply.is_file():
        raise FileNotFoundError('Scene PLY missing: ' + str(ply))
    if not source.is_dir():
        raise FileNotFoundError('3D saliency directory missing: ' + str(source))
    sys.path.insert(0, str(root))
    import util_gau
    xyz = np.asarray(util_gau.load_ply(str(ply)).xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < args.tiles:
        raise ValueError('Unexpected scene Gaussian XYZ: ' + str(xyz.shape))
    if not np.isfinite(xyz).all():
        raise ValueError('Scene has nonfinite coordinates')
    n = len(xyz)
    print(f'[SCENE] Persistent Gaussians: {n:,}, tiles={args.tiles}', flush=True)
    groups = spatial_tiles(xyz, args.tiles)
    centroids = np.array([xyz[ids].mean(axis=0) for ids in groups], dtype=np.float64)
    candidates = []
    for p in sorted(source.glob('*.npy')):
        m = re.fullmatch(r'(\d{3,6})', p.stem)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise RuntimeError('No numbered .npy saliency maps in ' + str(source))
    frame_scores = {}
    skipped = []
    suffix_aligned = []
    for frame, p in candidates:
        try:
            a = np.load(p, mmap_mode='r', allow_pickle=False)
            if a.ndim != 1 or not (n <= a.shape[0] <= n + args.max_extra):
                skipped.append((p.name, f'incompatible length {a.shape}'))
                continue
            if a.shape[0] != n:
                suffix_aligned.append((p.name, int(a.shape[0]-n)))
            v = np.asarray(a[-n:], dtype=np.float32)
            if not np.isfinite(v).all() or float(v.min()) < 0:
                skipped.append((p.name, 'nonfinite or negative saliency'))
                continue
            score = compute_scores(v, groups)
            if score is None:
                skipped.append((p.name, 'all-zero saliency'))
                continue
            frame_scores[frame] = score
        except Exception as exc:
            skipped.append((p.name, repr(exc)))
    if len(frame_scores) < 2:
        raise RuntimeError(f'Only {len(frame_scores)} usable maps; expected at least 2. Skips: {skipped[:12]}')
    frames = sorted(frame_scores)
    report = dict(
        format='qntc_dynamic_saliency_timeline_v1',
        description='Time-varying 3D SalFoM spatial importance; P0/P1 are rank proxies NOT camera agreement',
        scene_ply=str(ply), source_dir=str(source), points=n, tiles=args.tiles,
        frames=frames,
        scores=[frame_scores[k] for k in frames],
        centroids=centroids.tolist(),
        metric='top_1_percent_mean_per_tile_normalized_per_anchor',
        alignment='suffix of 3D per-Gaussian vector when additional Gaussian records are present',
        interpolation='linear per-tile importance between saved anchor frames; clamp outside anchor range',
        tier_rule='top2 P0, next2 P1, 4 nearest centroid tiles P2, remaining P3',
        skipped_maps=skipped[:100],
        suffix_aligned_maps=suffix_aligned[:100],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'[SALIENCY] Found={len(candidates)}, usable={len(frames)}, skipped={len(skipped)}', flush=True)
    print(f'[ANCHORS] {frames}', flush=True)
    if suffix_aligned:
        print(f'[ALIGNMENT] {len(suffix_aligned)} maps had extra entries; matched persistent Gaussian IDs by suffix, as in the previous audit.',flush=True)
    if skipped:
        for name, reason in skipped[:10]:
            print(f'[SKIPPED] {name}: {reason}', flush=True)
    print('[SOURCE] Saved 3D SalFoM, not new per-frame inference or per-camera overlap votes.', flush=True)
    print('[OUTPUT]', output,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--viewer', type=Path, default=Path.home()/'3DGS-QuaNTiC-Demo')
    ap.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    ap.add_argument('--saliency-dir', type=Path, default=DEFAULT_SALIENCY)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--tiles', type=int, default=32)
    ap.add_argument('--max-extra', type=int, default=100000,
                    help='Allow saliency arrays with up to N extra entries; use last N as in original experiment')
    args=ap.parse_args()
    if args.tiles < 8 or args.tiles > 256:
        ap.error('--tiles must be between 8 and 256')
    build(args)

if __name__ == '__main__':
    main()
