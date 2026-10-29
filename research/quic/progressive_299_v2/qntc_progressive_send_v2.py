#!/usr/bin/env python3
"""REAL QUIC sender: four separately committed partial INT4 streams per NTC.

Uses the existing project's aioquic transport. Orders blocks using the sampled
NTC114 tile-support map + changing 3D SalFoM tile scores when provided.
This is an approximate pilot ordering, NOT the exact gradient ranking measured
for NTC114 and NOT verified multiview agreement.
"""
import argparse
import asyncio
import json
import ssl
import tempfile
from pathlib import Path
import time
import torch
import numpy as np
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from quic_transport import ALPN, send_file
from qntc_progressive_wire import (HASH, TOTAL, extract_records_from_torch,
                                   encode_stage)

DEFAULT_BASE = Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat')
DEFAULT_AUDIT = Path.home()/'snap/firefox/common/NTC_tile_block_dependencies_114.json'
DEFAULT_TIMELINE = Path.home()/'snap/firefox/common/NTC_dynamic_saliency_timeline.json'


def get_weights(timeline_path, frame, n=32):
    if not timeline_path.is_file():
        raise FileNotFoundError('Dynamic 3D SalFoM timeline missing: '+str(timeline_path)+'; run prepare_dynamic_saliency.py')
    if not hasattr(get_weights, '_cache') or get_weights._cache[0] != str(timeline_path):
        t = json.loads(timeline_path.read_text())
        if t.get('tiles') != n or len(t['frames']) < 2:
            raise ValueError('bad SalFoM timeline')
        get_weights._cache = (str(timeline_path), t)
    t = get_weights._cache[1]
    frames = np.asarray(t['frames'], dtype=float)
    scores = np.asarray(t['scores'], dtype=float)
    frame = float(frame)
    current = np.array([np.interp(frame, frames, scores[:, i]) for i in range(n)])
    ranked = np.argsort(-current, kind='stable')
    tiers = np.full(n, .2, dtype=np.float64)
    tiers[ranked[:2]] = 12
    tiers[ranked[2:4]] = 4
    centres = np.asarray(t['centroids'], dtype=float)
    pool = [j for j in range(n) if j not in ranked[:4]]
    near = np.linalg.norm(centres[pool, None, :] - centres[ranked[:4]][None,:,:], axis=-1).min(axis=1)
    for j in np.asarray(pool)[np.argsort(near)[:4]]:
        tiers[j] = 1
    return tiers


def read_support(audit):
    if not audit.is_file():
        raise FileNotFoundError(f'Missing tile dependency audit: {audit}')
    d = json.loads(audit.read_text())
    if len(d['tiles']) != 32:
        raise ValueError('expected 32 spatial tiles')
    # The audit probes retained blocks of NTC114; other NTCs have changing IDs.
    return {int(k): np.asarray(v, dtype=np.int32)
            for k, v in d['retained_hash_block_to_tiles'].items()}


def ordered_records(records, weights, support, method):
    tail = [r for r in records if r[0] >= HASH]
    h = [r for r in records if r[0] < HASH]
    if method == 'original':
        return tail, h
    if method == 'random':
        rng = np.random.default_rng(42)
        rng.shuffle(h)
        return tail, h
    # Pilot support-weighted ordering: unknown blocks get neutral lowest score.
    # This reuses observed gradient support, NOT time-specific influence magnitude.
    def score(r):
        ts = support.get(r[0], [])
        return float(np.sum(weights[ts])) if len(ts) else 0.
    h.sort(key=lambda r: (-score(r), r[0]))
    return tail, h


async def send_payload(proto, relative, payload, tmpdir):
    path = Path(tmpdir)/'payload.qblk'
    path.write_bytes(payload)
    await send_file(proto, relative, path, timeout=120)


async def run(args):
    support = read_support(args.audit) if args.order == 'saliency' else {}
    conf = QuicConfiguration(is_client=True, alpn_protocols=ALPN)
    if args.insecure:
        conf.verify_mode = ssl.CERT_NONE
    elif args.ca:
        conf.load_verify_locations(str(args.ca))
    conf.server_name = args.host
    scene = args.scene.expanduser()
    wire = args.wire.expanduser()
    with tempfile.TemporaryDirectory(prefix='qntc_priority_quic_') as tmp:
        async with connect(args.host, args.port, configuration=conf) as proto:
            first = torch.load(wire / f'NTC_{args.start:06d}.pth',map_location='cpu',weights_only=True)
            bounds = {
                'xyz_bound_min': first['xyz_bound_min'].cpu().float().tolist(),
                'xyz_bound_max': first['xyz_bound_max'].cpu().float().tolist(),
            }
            bound_file = Path(tmp)/'bounds.json'
            bound_file.write_text(json.dumps(bounds))
            # Metadata arrives BEFORE initialization to avoid incorrect fallback bounds.
            await send_file(proto,'NTCs/progressive/bounds.json',bound_file)
            await send_file(proto,'init_3dgs.ply',scene/'init_3dgs.ply',timeout=120)
            await send_file(proto,'NTCs/config.json',scene/'NTCs/config.json',timeout=120)
            if args.warmup_delay > 0:
                print(f"[REAL QUIC] receiver warmup {args.warmup_delay:.1f}s",flush=True)
                await asyncio.sleep(args.warmup_delay)
            for frame in range(args.start,args.end+1):
                path=wire/f'NTC_{frame:06d}.pth'
                if not path.exists():
                    raise FileNotFoundError(path)
                state=torch.load(path,map_location='cpu',weights_only=True)
                records=extract_records_from_torch(state)
                weights=get_weights(args.timeline,frame) if args.order=='saliency' else np.ones(32)
                tail,h=ordered_records(records,weights,support,args.order)
                if len(tail)+len(h)!=1556:
                    raise ValueError(f'frame {frame}: invalid retained-record count '+
                                     f'HashGrid={len(h)} MLP={len(tail)}')
                if frame % 20 == 0 or frame == args.end:
                    print(f'[QUIC SENDER] frame={frame:03d}/{args.end:03d} '
                          f'HashGrid={len(h)} MLP={len(tail)}', flush=True)
                cut=[0,round(.25*len(h)),round(.5*len(h)),round(.75*len(h)),len(h)]
                for stage in range(4):
                    stage_records = (tail if stage==0 else []) + h[cut[stage]:cut[stage+1]]
                    payload=encode_stage(frame,stage,stage_records)
                    rel=f'NTCs/progressive/NTC_{frame:06d}_P{stage}.qblk'
                    t0=time.monotonic()
                    await send_payload(proto,rel,payload,tmp)
                    print(f'[QUIC SENT] frame={frame:03d} P{stage} records={len(stage_records)} bytes={len(payload)} commit_ms={1000*(time.monotonic()-t0):.1f}',flush=True)
                    if args.stage_delay > 0:
                        await asyncio.sleep(args.stage_delay)
            print('[QUIC SENDER COMPLETE] All requested updates committed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=5002)
    p.add_argument('--scene',type=Path,default=DEFAULT_BASE/'scenes/candidate')
    p.add_argument('--wire',type=Path,default=DEFAULT_BASE/'wire')
    p.add_argument('--audit',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--timeline',type=Path,default=DEFAULT_TIMELINE)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--end',type=int,default=298)
    p.add_argument('--order',choices=['saliency','original','random'],default='saliency')
    p.add_argument('--stage-delay',type=float,default=.045,help='demo: seconds between tier transfers (0.045 shows progressive changes; 0 hides them)')
    p.add_argument('--warmup-delay',type=float,default=2.0)
    p.add_argument('--ca',type=Path)
    p.add_argument('--insecure',action='store_true',help='only for loopback testing')
    a=p.parse_args()
    if not 0<=a.start<=a.end<=298 or a.stage_delay<0 or a.warmup_delay<0:
        p.error('0<=start<=end<=298 and stage-delay>=0 required')
    if not a.insecure and not a.ca:
        p.error('pass --ca cert.pem, or --insecure ONLY for local testing')
    asyncio.run(run(a))

if __name__=='__main__':
    main()
