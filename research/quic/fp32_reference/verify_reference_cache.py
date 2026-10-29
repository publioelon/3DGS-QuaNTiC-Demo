#!/usr/bin/env python3
"""Verify original scene files are byte-identical after native QUIC transfer."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from fp32_quic_sender import collect_files, DEFAULT_SCENE, DEFAULT_CACHE


def digest(path):
    hash_ = hashlib.sha256()
    with path.open('rb') as fh:
        for data in iter(lambda: fh.read(1024*1024), b''):
            hash_.update(data)
    return hash_.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=DEFAULT_SCENE)
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--frames',type=int,default=299)
    parser.add_argument('--sample',type=int,default=0,
                        help='0=verify all files, N>0=sample up to N frames plus first/last')
    args=parser.parse_args()
    source=collect_files(args.scene.expanduser().resolve(),args.frames)
    if args.sample:
        count=max(2,args.sample)
        indices={round(j*(args.frames-1)/(count-1)) for j in range(count)}
        indices.update({0,args.frames-1})
        keep={'init_3dgs.ply','NTCs/config.json'}
        for i in indices:
            keep.add(f'NTCs/NTC_{i:06d}.pth')
            keep.add(f'additional_3dgs/additions_{i:06d}.ply')
        source=[entry for entry in source if entry[0] in keep]
    cache=args.cache.expanduser().resolve()
    bad=[]
    for i,(rel,p) in enumerate(source,1):
        target=cache/rel
        if not target.is_file() or target.stat().st_size != p.stat().st_size:
            bad.append(rel + ': missing or different size')
        elif digest(p) != digest(target):
            bad.append(rel + ': SHA256 mismatch')
        if i%60==0 or i==len(source):
            print(f'[VERIFY] {i}/{len(source)}',flush=True)
    if bad:
        print('[FAIL] First mismatches:', *bad[:10],sep='\n',flush=True)
        raise SystemExit(2)
    print(f'[PASS] {len(source)} files byte-identical to reference scene',flush=True)


if __name__=='__main__':
    main()
