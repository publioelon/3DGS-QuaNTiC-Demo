#!/usr/bin/env python3
"""NTC114: full-population raw NTC output comparison by spatial tier and QUIC stage.

Safeguards: never writes to source scene, NTC, stage files, viewer or receiver.
GPU work is inference only; temporarily zeroes ONE block in a local model and restores it.
Reference is the COMPLETE sparse INT4 keep05 checkpoint (all four .qblk stages),
NOT the original FP32 NTC or rendered motion. Uses animated pre-NTC114 positions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MAGIC = b'QNP01'
HEADER = struct.Struct('!5sIBHH')
REC = struct.Struct('!If32s')
TOTAL = 31120
HASH = 30976
BLOCK = 64
N_EXPECT = 1556
COLORS = ['red', 'orange', 'yellow', 'blue']
P = ['P0', 'P1', 'P2', 'P3']
WEIGHTS = [12.0, 4.0, 1.0, 0.2]  # exploratory spatial-saliency weights


def balanced_tiles(xyz: np.ndarray, n: int = 32) -> list[np.ndarray]:
    """Exact partition rule in the user's original rank_ntc_frames.py."""
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


def load_stages(stage_dir: Path, frame: int):
    """Decode exact 4x QUIC INT4 file records, verifying full partition."""
    stages = []
    recs_by_id = {}
    stage_for = {}
    for prio in range(4):
        f = stage_dir / f'NTC_{frame:06d}_P{prio}.qblk'
        raw = f.read_bytes()
        if len(raw) < HEADER.size:
            raise RuntimeError(f'Truncated .qblk: {f}')
        magic, found_frame, found_p, expected, count = HEADER.unpack_from(raw)
        if (magic, found_frame, found_p, expected) != (MAGIC, frame, prio, N_EXPECT):
            raise RuntimeError(f'Unexpected .qblk header in {f}: {magic,found_frame,found_p,expected}')
        if len(raw) != HEADER.size + count * REC.size:
            raise RuntimeError(f'Bad .qblk record length in {f}')
        this = []
        for i in range(count):
            bid, scale, packed = REC.unpack_from(raw, HEADER.size + i * REC.size)
            if bid >= TOTAL or not np.isfinite(scale) or scale < 0 or bid in recs_by_id:
                raise RuntimeError(f'Bad or repeated ID {bid} in {f}')
            q = np.frombuffer(packed, dtype=np.uint8)
            vals = np.empty(64, dtype=np.float32)
            vals[0::2] = (q & 15).astype(np.float32) - 8
            vals[1::2] = (q >> 4).astype(np.float32) - 8
            vals *= np.float32(scale)
            recs_by_id[bid] = vals
            stage_for[bid] = prio
            this.append(bid)
        stages.append(this)
    if len(recs_by_id) != N_EXPECT:
        raise RuntimeError(f'Expected {N_EXPECT} distinct weight blocks; got {len(recs_by_id)}')
    return stages, recs_by_id, stage_for


def full_params(ids, records):
    out = np.zeros((TOTAL, BLOCK), dtype=np.float32)
    for bid in ids:
        out[bid] = records[bid]
    return out.reshape(-1)


def choose_fixed(ids: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(ids) <= limit:
        return np.asarray(ids, dtype=np.int64)
    r = np.random.default_rng(seed)
    return np.sort(r.choice(ids, size=limit, replace=False)).astype(np.int64)


def rms(v: np.ndarray) -> float:
    if not len(v):
        return float('nan')
    return float(np.sqrt(np.mean(np.sum(np.square(v.astype(np.float64)), axis=1))))


def get_scores(pred: np.ndarray, ref: np.ndarray) -> dict:
    dt = pred[:, :3].astype(np.float64) - ref[:, :3].astype(np.float64)
    dr = pred[:, 3:7].astype(np.float64) - ref[:, 3:7].astype(np.float64)
    t_ref = rms(ref[:, :3]); r_ref = rms(ref[:, 3:7])
    t = rms(dt); r = rms(dr)
    return {'translation_raw_rms_error': t, 'rotation_raw_rms_error': r,
            'translation_raw_ref_rms': t_ref, 'rotation_raw_ref_rms': r_ref,
            'translation_rel_rms_error': t / max(t_ref, 1e-12),
            'rotation_rel_rms_error': r / max(r_ref, 1e-12)}


def evaluate(model, x: 'torch.Tensor', batch: int = 4096):
    """Tensor already on GPU, coordinates normalized to checkpoint bounds.
    Pad to 128 rows for versions of tiny-cuda-nn requiring batch granularity.
    """
    import torch
    ys = []
    with torch.no_grad():
        for beg in range(0, len(x), batch):
            z = x[beg:beg + batch].contiguous()
            n = len(z)
            if n % 128:
                z = torch.cat((z, z[-1:].expand(128 - n % 128, -1)), dim=0)
            out = model(z)[:n, :7].float()
            ys.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(ys, axis=0) if ys else np.empty((0,7), np.float32)


def make_gpu_points(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray, device):
    import torch
    coords = ((xyz.astype(np.float32) - lo) / (hi - lo)).astype(np.float32)
    return torch.as_tensor(coords, dtype=torch.float32, device=device)


def gather_points(x_all, gids):
    import torch
    ids = torch.as_tensor(np.asarray(gids, dtype=np.int64), device=x_all.device)
    return x_all.index_select(0, ids)


def stage_rows(model, stage_groups: list[list[int]], weights: dict[int,np.ndarray],
               x_all, stage_gids: list[np.ndarray], full_y_stage: list[np.ndarray],
               batch: int, method: str):
    """Actual multi-block partial network, not a sum of independent ablations."""
    import torch
    current = np.zeros((TOTAL, 64), dtype=np.float32)
    rows = []
    for st, ids in enumerate(stage_groups):
        for bid in ids:
            current[bid] = weights[bid]
        with torch.no_grad():
            model.params.copy_(torch.from_numpy(current.reshape(-1)).to(device=model.params.device,dtype=model.params.dtype))
        for tier in range(4):
            gids = stage_gids[tier]
            p = evaluate(model, gather_points(x_all,gids), batch=batch)
            s = get_scores(p, full_y_stage[tier])
            rows.append({'ordering': method, 'received_through': f'P{st}',
                         'spatial_priority': f'P{tier}', 'color': COLORS[tier],
                         'evaluated_gaussians': len(gids),
                         'cumulative_blocks': sum(map(len, stage_groups[:st+1])), **s})
        print(f'[STAGE] {method}: through P{st} blocks={sum(map(len, stage_groups[:st+1]))} '
              f'red raw-translation RMS={rows[-4]["translation_raw_rms_error"]:.8g}', flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('[SAVED]', path, flush=True)


def main():
    home = Path.home()
    root = home / 'snap/firefox/common'
    base = root / 'QNTC_dynamic_299_priorities_v1'
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--frame', type=int, default=114)
    pa.add_argument('--stage-dir', type=Path, default=base/'stages')
    pa.add_argument('--order', type=Path, default=base/'orders/NTC_000114.npz')
    pa.add_argument('--scene', type=Path, default=home/'qntc_scenes/flame_steak_official')
    pa.add_argument('--viewer', type=Path, default=home/'3DGS-QuaNTiC-Demo')
    pa.add_argument('--state', type=Path, default=root/'QNTC_ref_state_114/state_000114.pt')
    pa.add_argument('--mapping', type=Path, default=root/'QNTC_ntc114_animated_map_v2/ntc_114_hash_block_to_gaussian_ids.npz')
    pa.add_argument('--output', type=Path, default=root/'QNTC_ntc114_full_population_v3')
    pa.add_argument('--samples-per-tile', type=int, default=128,
                    help='Stratified stagewise comparison samples per each of 32 tiles.')
    pa.add_argument('--per-block-per-color', type=int, default=64,
                    help='For individual ablation, sampled Gaussians in each touched color.')
    pa.add_argument('--max-blocks', type=int, default=48,
                    help='0 means ALL 1,547 retained HashGrid + 9 MLP blocks; default safe pilot 48.')
    pa.add_argument('--batch', type=int, default=4096)
    pa.add_argument('--numeric-threshold', type=float, default=1e-6,
                    help='Raw-output RMS threshold used only to label measurable effects.')
    pa.add_argument('--compare-reranked', action='store_true',
                    help='Full ablation required: compare priorities by coverage-adjusted, color-weighted translation and rotation effects.')
    pa.add_argument('--rerank-csv', type=Path, default=root/'QNTC_ntc114_color_motion_v2/motion_rerank_original_block_ids.csv')
    pa.add_argument('--self-test', action='store_true', help='CPU-only basic reader/metric checks.')
    a = pa.parse_args()
    if a.self_test:
        toy = np.zeros((4,7),dtype=np.float32)
        toy[:,0]=2
        assert abs(get_scores(toy,np.zeros_like(toy))['translation_raw_rms_error']-2)<1e-9
        assert sum(len(x) for x in balanced_tiles(np.arange(96,dtype=np.float32).reshape(32,3),8))==32
        print('[PASS] CPU-only scoring and tile split selftest')
        return
    if (a.frame != 114 or a.samples_per_tile < 8 or a.per_block_per_color < 4
            or a.max_blocks < 0 or a.batch < 128 or a.batch % 128 != 0):
        pa.error('This validated experiment is NTC114 only; check positive samples, batch divisible by 128.')
    import torch
    try:
        import tinycudann as tcnn
    except Exception as e:
        raise RuntimeError('tinycudann must be installed in qntcstream') from e
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; run on the qntcstream laptop')
    for src in (a.scene/'init_3dgs.ply', a.scene/'NTCs/config.json',
                a.state, a.mapping, a.order):
        if not src.is_file():
            raise FileNotFoundError(f'Needed input missing: {src}')
    a.output.mkdir(parents=True, exist_ok=True)
    stages, records, stage_for = load_stages(a.stage_dir, 114)
    stages_hash = [[bid for bid in stage if bid < HASH] for stage in stages]
    all_hash = set(bid for stage in stages_hash for bid in stage)
    print('[QUIC] stage sizes:', [len(s) for s in stages],
          'HashGrid=', len(all_hash), 'MLP=', N_EXPECT - len(all_hash), flush=True)

    # The original ranker uses INITIAL xyz to assign 32 fixed spatial colors.
    # The V2 structural mapping and the numerical evaluation use ANIMATED xyz.
    sys.path.insert(0, str(a.viewer.resolve()))
    import util_gau
    init_xyz = np.asarray(util_gau.load_ply(str(a.scene/'init_3dgs.ply')).xyz,
                          dtype=np.float32)
    state = torch.load(a.state, map_location='cpu', weights_only=True)
    if 'timestep' not in state or int(state['timestep']) != 114:
        raise RuntimeError('State must explicitly identify timestep=114')
    for k in ('xyz','rot'):
        if k not in state:
            raise RuntimeError(f'Missing animated state field {k}')
    value = state['xyz']
    if isinstance(value, torch.Tensor):
        animated_xyz = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
    else:
        animated_xyz = np.asarray(value, dtype=np.float32)
    animated_xyz = np.ascontiguousarray(animated_xyz)
    if animated_xyz.shape != (len(init_xyz), 3) or not np.isfinite(animated_xyz).all():
        raise RuntimeError(f'Invalid animated.xyz shape {animated_xyz.shape}')
    sha = hashlib.sha256(animated_xyz.tobytes()).hexdigest()
    fingerprint_file = a.mapping.parent / 'positions_sha256.txt'
    if fingerprint_file.is_file():
        val = re.search(r'\b[a-fA-F0-9]{64}\b', fingerprint_file.read_text())
        if val and val.group(0).lower() != sha:
            raise RuntimeError('Animated XYZ SHA256 does not match the V2 mapping. Stopping.')
    print('[POSITIONS] animated pre-NTC114 Gaussians:', len(animated_xyz),
          'SHA256:', sha, flush=True)
    print('[SPATIAL] colors use INITIAL XYZ partition, but NTC evaluation uses ANIMATED XYZ',flush=True)

    with np.load(a.mapping, allow_pickle=False) as d:
        map_bids = np.asarray(d['block_ids'], dtype=np.int32)
        offsets = np.asarray(d['offsets'], dtype=np.int64)
        gaussians = np.asarray(d['gaussian_ids'], dtype=np.int64)
        inside = np.asarray(d['inside_bounds_ids'], dtype=np.int64)
        gaussian_count = int(d['gaussian_count'])
    if len(map_bids) != len(all_hash) or set(map(int,map_bids)) != all_hash:
        raise RuntimeError('Retained HashGrid IDs in animated map differ from QUIC stage records')
    if (len(offsets) != len(map_bids)+1 or offsets[0] != 0 or offsets[-1] != len(gaussians)
            or len(init_xyz) != gaussian_count or np.any(gaussians < 0)
            or np.any(gaussians >= gaussian_count)):
        raise RuntimeError('Corrupted/incompatible Gaussian dependency map')
    if len(np.unique(inside)) != len(inside):
        raise RuntimeError('Duplicate in-bounds IDs')

    with np.load(a.order, allow_pickle=False) as order:
        tiers = np.asarray(order['tile_tiers'], dtype=np.uint8)
        scheduled_ids = np.asarray(order['ordered_block_ids'], dtype=np.int64)
        wire_path = Path(str(order['source_wire'].item()))
    if not wire_path.is_file():
        raise FileNotFoundError(f'Checkpoint for exact bounds is absent: {wire_path}')
    if len(tiers) != 32 or (tiers > 3).any() or sum(map(len, stages)) != len(scheduled_ids):
        raise RuntimeError('Invalid scheduled tiles or record counts')
    if not np.array_equal(np.asarray([j for part in stages for j in part],dtype=np.int64),
                          scheduled_ids):
        raise RuntimeError('QUIC stage order does not match saved frame 114 schedule')
    source = torch.load(wire_path, map_location='cpu', weights_only=True)
    lo = np.asarray(source['xyz_bound_min'],dtype=np.float32).reshape(1,3)
    hi = np.asarray(source['xyz_bound_max'],dtype=np.float32).reshape(1,3)
    if not np.all(hi>lo):
        raise RuntimeError('Invalid NTC normalization bounds')
    normal = (animated_xyz - lo) / (hi - lo)
    measured_inside = np.flatnonzero(np.logical_and(normal >= 0, normal <= 1).all(axis=1))
    if not np.array_equal(np.sort(inside), measured_inside):
        raise RuntimeError('Saved V2 inside-bounds Gaussian IDs mismatch this animated state/bounds')
    tiles = balanced_tiles(init_xyz, 32)
    gaussian_color = np.full(len(init_xyz), 255, dtype=np.uint8)
    for i, gids in enumerate(tiles):
        gaussian_color[gids] = tiers[i]
    if np.any(gaussian_color>3):
        raise RuntimeError('Not all Gaussians assigned to spatial colors')
    # Confirm previous exact structural crosswalk counts to catch accidental mismatches.
    seen_color = [set() for _ in range(4)]
    for i, bid in enumerate(map_bids):
        touched = np.unique(gaussian_color[gaussians[offsets[i]:offsets[i+1]]])
        for t in touched:
            seen_color[int(t)].add(int(bid))
    counts = [len(s) for s in seen_color]
    if counts != [1056,1026,1097,1466]:
        raise RuntimeError('Spatial-color/mapping crosswalk changed: '
                           f'{counts} (expected 1056,1026,1097,1466); stopping')
    print('[CHECK] Animated map + original tile colors agree with prior audit:', counts,flush=True)

    cfg = json.loads((a.scene/'NTCs/config.json').read_text())
    device = torch.device('cuda')
    model = tcnn.NetworkWithInputEncoding(n_input_dims=3,n_output_dims=8,
                encoding_config=cfg['encoding'], network_config=cfg['network']).to(device)
    model.eval()
    if model.params.numel() != TOTAL*BLOCK:
        raise RuntimeError(f'Unexpected tiny-cuda-nn parameter count: {model.params.numel()}')
    x_all = make_gpu_points(animated_xyz, lo, hi, device)
    full_ids = [bid for stage in stages for bid in stage]
    full_cpu = full_params(full_ids, records)
    with torch.no_grad():
        model.params.copy_(torch.from_numpy(full_cpu).to(device=device, dtype=model.params.dtype))
    print('[GPU] model params=', model.params.numel(), 'dtype=', model.params.dtype,
          'CUDA device=',torch.cuda.get_device_name(0), flush=True)
    # Full sparse reference outputs for EVERY in-bounds persistent Gaussian.
    ref_y = np.zeros((len(init_xyz), 7),dtype=np.float32)
    for s in range(0,len(measured_inside),a.batch):
        g = measured_inside[s:s+a.batch]
        ref_y[g] = evaluate(model,gather_points(x_all,g),batch=a.batch)
    print('[REFERENCE] Full sparse INT4 NTC evaluated at',len(measured_inside),
          'animated Gaussian positions',flush=True)

    available_mask = np.zeros(len(init_xyz), dtype=bool)
    available_mask[measured_inside] = True
    stage_gids = [np.flatnonzero(available_mask & (gaussian_color == t)) for t in range(4)]
    print('[FULL POPULATION] By color:',dict(zip(P,map(len,stage_gids))),flush=True)
    full_y_stage = [ref_y[g].copy() for g in stage_gids]
    color_population = [int(np.count_nonzero(available_mask & (gaussian_color == t))) for t in range(4)]
    color_reference_translation_rms = [rms(y[:,:3]) for y in full_y_stage]
    color_reference_rotation_rms = [rms(y[:,3:7]) for y in full_y_stage]
    print('[IN-BOUNDS COLOR POPULATIONS]',dict(zip(P,color_population)),flush=True)
    print('[RANKING] Coverage-adjusted, color-wide RMS estimates using sampled affected Gaussians.',flush=True)
    current = stage_rows(model,stages,records,x_all,stage_gids,full_y_stage,a.batch,'existing_QUIC')
    write_csv(a.output/'stagewise_raw_ntc_motion_errors.csv', current)

    if a.compare_reranked:
        if not a.rerank_csv.is_file():
            raise FileNotFoundError(f'Missing V2 ranking: {a.rerank_csv}')
        groups = [[] for _ in range(4)]
        with a.rerank_csv.open(newline='') as f:
            for row in csv.DictReader(f):
                bid = int(row['block_id'])
                pr = int(row['experimental_priority'][1:])
                if pr not in range(4):
                    raise RuntimeError(f'Invalid ranking stage {pr}')
                groups[pr].append(bid)
        if [len(x) for x in groups] != [len(x) for x in stages] or sorted(b for g in groups for b in g) != sorted(full_ids):
            raise RuntimeError('V2 ranking must contain all original 1556 blocks and preserve stage capacities')
        alt = stage_rows(model,groups,records,x_all,stage_gids,full_y_stage,a.batch,'experimental_motion_ranking')
        write_csv(a.output/'full_population_existing_vs_motion.csv',current+alt)
    report = {
        'frame':114, 'reference':'all 1556 sparse INT4 blocks',
        'total_persistent_gaussians':len(init_xyz),
        'in_bounds_evaluated_gaussians':len(measured_inside),
        'color_population':dict(zip(P,color_population)),
        'animated_xyz_sha256':sha,
        'raw_outputs':'translation channels 0:3 and rotation channels 3:7; rotation is NOT an angular error',
        'notes':'All in-bounds persistent Gaussians, no added Gaussians, no rendered images. '
                'Reference is full sparse INT4 NTC, not original FP32 NTC. '
                'No source files or QUIC stages modified.',
        'compared_reranking':bool(a.compare_reranked)
    }
    meta=a.output/'full_population_report.json'
    meta.write_text(json.dumps(report,indent=2)+'\n')
    print('[SAVED]',meta,flush=True)
    print('[SAFE] Offline evaluation only; no source or QUIC stages changed.',flush=True)

if __name__=='__main__':
    main()
