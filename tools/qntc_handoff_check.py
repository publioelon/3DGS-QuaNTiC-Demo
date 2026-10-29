#!/usr/bin/env python3
"""No-dependency checks for 3DGS-QuaNTiC handoff and QNP01 stage structure."""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import struct
import sys

HEADER = struct.Struct("!5sIBHH")
RECORD = struct.Struct("!If32s")


def read_stage(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated stage header")
    magic, frame, priority, total, count = HEADER.unpack_from(data)
    if magic != b"QNP01" or priority not in range(4) or not (0 < total <= 31120):
        raise ValueError("invalid QNP01 stage")
    if len(data) != HEADER.size + count * RECORD.size:
        raise ValueError("payload length inconsistent with record count")
    ids = []
    for i in range(count):
        block, scale, packed = RECORD.unpack_from(data, HEADER.size + i * RECORD.size)
        if block >= 31120 or not (scale >= 0 and scale < float("inf")):
            raise ValueError("bad block ID or scale")
        ids.append(block)
    if len(ids) != len(set(ids)) or count > total:
        raise ValueError("duplicate IDs or impossible count")
    return frame, priority, total, count


def doctor():
    root = Path(__file__).resolve().parents[1]
    viewer = Path.home() / "3DGS-QuaNTiC-Demo"
    print("3DGS-QuaNTiC · environment check")
    print("Python", sys.version.split()[0])
    for name in ("numpy", "torch", "tinycudann", "aioquic", "glfw"):
        try:
            ok = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            ok = False
        print(f"{name:12} {'OK' if ok else 'MISSING'}")
    files = [viewer/"main.py", viewer/"renderer_cuda.py",
             viewer/"qntc_progressive_live.py",
             Path.home()/"qntc_scenes/flame_steak_official/init_3dgs.ply"]
    for path in files:
        print(f"{'OK' if path.is_file() else 'MISSING':8} {path}")
    if (viewer/"main.py").is_file():
        print("QUIC hook:", "present" if "quic_listen" in (viewer/"main.py").read_text(errors="replace")
              else "absent; local viewer may be newer")
    print("Source only. This does not modify the viewer or train weights.")


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("doctor")
    q = s.add_parser("check-stage")
    q.add_argument("path", type=Path)
    a = p.parse_args()
    if a.cmd == "doctor":
        doctor()
    else:
        frame, stage, total, count = read_stage(a.path)
        print(f"PASS frame={frame} P{stage} blocks={count}/{total} bytes={a.path.stat().st_size}")


if __name__ == "__main__":
    main()
