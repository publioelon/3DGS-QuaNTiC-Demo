#!/usr/bin/env python3
"""Reference-fidelity QUIC sender for native 3DGS-QuaNTiC receiver.

Transfers ORIGINAL FP32 NTC checkpoint files and matching per-frame Gaussian
additions unchanged. No quantization, priority blocks, or receiver patches.

To guarantee first network NTC bounds are read from the original checkpoint,
NTC_000000.pth is transferred before config.json (which creates the network).
The default full prebuffer holds back config until every requested frame has
been committed, yielding a deterministic file-over-QUIC reference test.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import ssl
from pathlib import Path

DEFAULT_SCENE = Path.home() / 'qntc_scenes/flame_steak_official'
DEFAULT_CACHE = Path('/tmp/qntc_fp32_reference_cache')


def collect_files(scene: Path, frames: int):
    if not 1 <= frames <= 299:
        raise ValueError('frames must be between 1 and 299 NTC updates')
    required = [
        ('init_3dgs.ply', scene / 'init_3dgs.ply'),
        ('NTCs/config.json', scene / 'NTCs' / 'config.json'),
    ]
    for idx in range(frames):
        required.extend([
            (f'additional_3dgs/additions_{idx:06d}.ply',
             scene / 'additional_3dgs' / f'additions_{idx:06d}.ply'),
            (f'NTCs/NTC_{idx:06d}.pth', scene / 'NTCs' / f'NTC_{idx:06d}.pth'),
        ])
    absent = [str(p) for _, p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError(f'Missing {len(absent)} scene files; first missing: {absent[:6]}')
    return required


def send_order(required, frames, prebuffer):
    """Publish config only after at least prebuffer complete NTCs and additions."""
    if not 1 <= prebuffer <= frames:
        raise ValueError('prebuffer must be from 1 to frames')
    source = dict(required)
    order = [('init_3dgs.ply', source['init_3dgs.ply'])]
    for i in range(frames):
        a = f'additional_3dgs/additions_{i:06d}.ply'
        n = f'NTCs/NTC_{i:06d}.pth'
        order.append((a, source[a]))
        order.append((n, source[n]))
        if i + 1 == prebuffer:
            order.append(('NTCs/config.json', source['NTCs/config.json']))
    return order


def nearest_existing(path: Path):
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def preflight(required, cache: Path, safety: float):
    total_bytes = sum(p.stat().st_size for _, p in required)
    available = os.statvfs(nearest_existing(cache))
    free = available.f_bavail * available.f_frsize
    needed = int(total_bytes * safety + 128 * 1024 * 1024)
    print(f'[PREFLIGHT] {len(required)} source files, {total_bytes / 2**30:.3f} GiB', flush=True)
    print(f'[PREFLIGHT] Cache destination: {cache}', flush=True)
    print(f'[PREFLIGHT] Free on destination filesystem: {free / 2**30:.3f} GiB', flush=True)
    print(f'[PREFLIGHT] Required free (safety {safety:.2f} + 128 MiB): {needed / 2**30:.3f} GiB', flush=True)
    source_root = required[0][1].parent.resolve()
    if (cache.name != 'qntc_fp32_reference_cache' or cache == source_root or
            cache.is_relative_to(source_root)):
        raise RuntimeError('Cache must be a dedicated directory named qntc_fp32_reference_cache, outside the source scene')
    if free < needed:
        raise RuntimeError('Insufficient free disk space. Do not start the receiver or sender. '
                           'Use a different --cache and matching receiver QNTC_FP32_CACHE.')
    return total_bytes


async def transmit(args, order, total_bytes):
    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration
    from quic_transport import ALPN, send_file

    if args.host not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('This reference pilot permits local loopback only')
    config = QuicConfiguration(is_client=True, alpn_protocols=ALPN)
    config.verify_mode = ssl.CERT_NONE  # local loopback pilot only
    config.server_name = 'localhost'
    committed = 0
    async with connect(args.host, args.port, configuration=config) as conn:
        for i, (relative, source) in enumerate(order, 1):
            await send_file(conn, relative, source, timeout=args.timeout)
            committed += source.stat().st_size
            if relative == 'NTCs/config.json':
                print('[QUIC] NTC/config received after prebuffer. Viewer can initialize '
                      'with exact original checkpoint bounds.', flush=True)
            if relative.startswith('NTCs/NTC_') and relative.endswith('.pth'):
                frame = int(Path(relative).stem.split('_')[-1])
                if frame in {0, 1, 9, 29, args.frames-1} or (frame+1) % 30 == 0:
                    print(f'[QUIC] NTC {frame:03d}/{args.frames-1:03d}, '
                          f'confirmed bytes={committed/2**20:.1f} MiB', flush=True)
        print(f'[QUIC COMPLETE] {i} files confirmed by receiver; '
              f'{committed/2**30:.3f} GiB transmitted.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--frames', type=int, default=299,
                        help='number of NTCs/additions, default full 299')
    parser.add_argument('--prebuffer', type=int, default=None,
                        help='how many updates to receive before viewer can start; default all')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5003)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--disk-safety', type=float, default=1.10)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.scene = args.scene.expanduser().resolve()
    args.cache = args.cache.expanduser().resolve()
    if args.prebuffer is None:
        args.prebuffer = args.frames
    if not (1.0 <= args.disk_safety <= 2.0):
        parser.error('disk-safety must be in [1.0, 2.0]')
    required = collect_files(args.scene, args.frames)
    order = send_order(required, args.frames, args.prebuffer)
    total = preflight(required, args.cache, args.disk_safety)
    print(f'[PLAN] {args.frames} updates, prebuffer={args.prebuffer}, '
          f'first={order[0][0]}, last={order[-1][0]}',flush=True)
    if args.preflight_only:
        print('[READY] Source exists and disk space sufficient. Nothing transmitted.', flush=True)
        return
    asyncio.run(transmit(args, order, total))


if __name__ == '__main__':
    main()
