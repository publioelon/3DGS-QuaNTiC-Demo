#!/usr/bin/env python3
"""Q3DGStream commands that work before installing the optional GPU stack.

Usage:
    python3 q3dgstream.py demo
    python3 q3dgstream.py doctor
    python3 q3dgstream.py test
    python3 q3dgstream.py inspect-stage PATH
    python3 q3dgstream.py viewer [viewer arguments...]
"""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
HEADER = struct.Struct('!5sIBHH')
RECORD = struct.Struct('!If32s')


def parse_qblk(data: bytes) -> tuple[int, int, int, list[tuple[int, float, bytes]]]:
    """Parse QNP01 header + records; validate lengths, IDs and scales."""
    if len(data) < HEADER.size:
        raise ValueError('truncated QNP01 stage header')
    magic, frame, stage, total, count = HEADER.unpack_from(data)
    if magic != b'QNP01':
        raise ValueError('invalid magic: expected QNP01')
    if not 0 <= stage <= 3 or total == 0 or count > total:
        raise ValueError('invalid priority or block counts')
    if len(data) != HEADER.size + count * RECORD.size:
        raise ValueError('invalid stage length')
    records=[]
    ids=set()
    for i in range(count):
        block, scale, packed = RECORD.unpack_from(data,HEADER.size + i*RECORD.size)
        if block >= 31120 or block in ids or not math.isfinite(scale) or scale < 0:
            raise ValueError(f'invalid block ID or quantization scale at record {i}')
        ids.add(block)
        records.append((block,scale,packed))
    return frame,stage,total,records


def demo() -> None:
    """No-dependency demonstration of the project-defined QNP01 stage format."""
    # 64 INT4 values packed into 32 bytes: alternate q=+1 and q=0.
    packed=bytes([0x89] * 32)
    raw=HEADER.pack(b'QNP01', 114, 0, 1556, 1) + RECORD.pack(93, 0.25, packed)
    frame,stage,total,records=parse_qblk(raw)
    print('Q3DGStream · QNP01 decoder demo')
    print(f'  Frame: {frame}   Priority: P{stage}   Blocks: {len(records)} / {total}')
    print(f'  Block ID: {records[0][0]}   Scale: {records[0][1]}')
    print(f'  First packed byte: 0x{records[0][2][0]:02X} → q=(+1, 0) → w=(0.25, 0.0)')
    print(f'  Encoded stage size: {len(raw)} bytes')
    print('PASS: QNP01 header and INT4 record parsed successfully')


def doctor() -> None:
    print('Q3DGStream · environment check')
    print('  Python:',sys.version.split()[0])
    for module in ('numpy','torch','tinycudann','aioquic','glfw','OpenGL'):
        try: ok=importlib.util.find_spec(module) is not None
        except (ImportError,ModuleNotFoundError,ValueError): ok=False
        print(f'  {module:12} {"available" if ok else "not installed"}')
    for label,path in [('Viewer',ROOT/'main.py'),('NTC renderer',ROOT/'renderer_cuda.py'),
                       ('Flame Steak scene',Path.home()/'qntc_scenes/flame_steak_official/init_3dgs.ply'),
                       ('Local QUIC receiver',ROOT/'qntc_progressive_live.py')]:
        print(f'  {label:20} {"OK" if path.is_file() else "not found"}: {path}')
    if (ROOT/'main.py').exists():
        print('  QUIC viewer hook:', 'available' if 'quic_listen' in (ROOT/'main.py').read_text(errors='replace') else 'not integrated in this checkout')


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='action',required=True)
    s.add_parser('demo',help='demo QNP01 with no additional dependencies')
    s.add_parser('doctor',help='check optional dependencies and scene files')
    s.add_parser('test',help='run CPU source/format tests')
    stage=s.add_parser('inspect-stage',help='validate a .qblk file and show frame/stage/block count')
    stage.add_argument('path',type=Path)
    viewer=s.add_parser('viewer',help='start original dynamic 3DGS viewer with given arguments')
    viewer.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=='demo': demo()
    elif a.action=='doctor':doctor()
    elif a.action=='test': return subprocess.call([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT)
    elif a.action=='inspect-stage':
        frame,idx,total,records=parse_qblk(a.path.read_bytes())
        print(f'PASS  frame={frame}  P{idx}  blocks={len(records)}/{total}  bytes={a.path.stat().st_size}')
    elif a.action=='viewer': return subprocess.call([sys.executable,str(ROOT/'main.py'),*a.args],cwd=ROOT)
    return 0

if __name__=='__main__':raise SystemExit(main())
