#!/usr/bin/env python3
"""Q3DGStream CUDA and sparse-NTC smoke test. Never modifies the working viewer."""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import time

HEADER = struct.Struct("!5sIBHH")
RECORD = struct.Struct("!If32s")
HOME = Path.home()
CASE = Path("/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/"
            "lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat")
STAGES = HOME / "snap/firefox/common/QNTC_dynamic_299_priorities_v1/stages"
STATES = HOME / "snap/firefox/common/QNTC_299_student_states"
SCENE = HOME / "qntc_scenes/flame_steak_official"

def stage_check(frame: int = 114) -> str:
    all_ids = set()
    sizes = []
    for priority in range(4):
        path = STAGES / f"NTC_{frame:06d}_P{priority}.qblk"
        data = path.read_bytes()
        if len(data) < HEADER.size:
            raise ValueError(f"Short stage header: {path}")
        magic, idx, part, total, count = HEADER.unpack_from(data)
        if (magic, idx, part, total) != (b"QNP01", frame, priority, 1556):
            raise ValueError(f"Incorrect stage header: {path}")
        if len(data) != HEADER.size + count * RECORD.size:
            raise ValueError(f"Bad file size: {path}")
        for i in range(count):
            block_id, scale, packed = RECORD.unpack_from(data, HEADER.size + i * RECORD.size)
            if block_id >= 31120 or block_id in all_ids or not math.isfinite(scale) or scale < 0:
                raise ValueError(f"Invalid record in: {path}")
            all_ids.add(block_id)
        sizes.append(len(data))
    if len(all_ids) != 1556:
        raise ValueError(f"Expected 1556 unique blocks; got {len(all_ids)}")
    return f"frame={frame}, 1556 unique blocks, {sum(sizes):,} staged bytes"

def cuda_ntc_test() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA unavailable; check driver, virtualenv and installation")
    import tinycudann as tcnn
    gpu = torch.cuda.get_device_name(0)
    config = json.loads((SCENE / "NTCs/config.json").read_text())
    # Load the true dequantized INT4 keep05 model and its validated input state.
    runtime = CASE / "runtime/NTC_000114.pth"
    wire = CASE / "wire/NTC_000114.pth"
    state = STATES / "state_000114.pt"
    ckpt = torch.load(runtime, map_location="cpu", weights_only=True)
    source = torch.load(wire, map_location="cpu", weights_only=True)
    gaussian = torch.load(state, map_location="cpu", weights_only=True)
    params = ckpt["model.params"].reshape(-1)
    if params.numel() != 1991680:
        raise ValueError(f"Unexpected NTC parameter count: {params.numel()}")
    net = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=8,
        encoding_config=config["encoding"], network_config=config["network"]
    ).cuda().eval()
    if net.params.numel() != params.numel():
        raise ValueError("NTC architecture does not match checkpoint")
    with torch.no_grad():
        net.params.copy_(params.to(device="cuda", dtype=net.params.dtype))
        xyz = gaussian["xyz"].float()
        low = source["xyz_bound_min"].float().reshape(1, 3)
        high = source["xyz_bound_max"].float().reshape(1, 3)
        normalized = (xyz - low) / (high - low)
        active = ((normalized >= 0) & (normalized <= 1)).all(dim=1)
        x = normalized[active][:1024].to("cuda").contiguous()
        if x.shape != (1024, 3):
            raise ValueError(f"Not enough active Gaussians: {x.shape}")
        torch.cuda.synchronize()
        start = time.monotonic()
        y = net(x).float()
        torch.cuda.synchronize()
        duration_ms = (time.monotonic() - start) * 1000
        if y.shape != (1024, 8) or not bool(torch.isfinite(y).all().item()):
            raise ValueError("Nonfinite or incorrectly shaped NTC output")
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return (f"{gpu}, 1,991,680 loaded weights, 1024 actual Gaussian "
            f"positions, {duration_ms:.2f} ms forward, peak {peak:.1f} MiB CUDA")

def viewer_hooks(viewer: Path) -> str:
    files = [viewer / p for p in ("main.py", "renderer_cuda.py",
                                  "qntc_progressive_live.py", "quic_transport.py")]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError("Missing local QUIC receiver files: " + ", ".join(missing))
    src = (viewer / "main.py").read_text(errors="replace")
    if "--quic_listen" not in src and "quic_listen" not in src:
        raise RuntimeError("The local main.py lacks --quic_listen support")
    return "receiver, CUDA renderer, progressive sidecar, QUIC transport present"

def run_subprocess(name: str, command: list[str], cwd: Path) -> str:
    print(f"\n[RUN] {name}: {' '.join(command)}", flush=True)
    code = subprocess.call(command, cwd=cwd)
    if code:
        raise RuntimeError(f"{name} failed with exit code {code}")
    return "completed successfully"

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--viewer", type=Path, default=HOME / "3DGS-QuaNTiC-Demo",
                   help="Locally patched working viewer (not the public baseline checkout)")
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Git checkout containing research/evaluation")
    p.add_argument("--verify-all", action="store_true", help="Verify 299 stages against wire checkpoints")
    p.add_argument("--evaluate-three", action="store_true", help="Evaluate frames 113 through 115 on CUDA")
    p.add_argument("--check-quic", action="store_true", help="Require local QUIC receiver hooks")
    p.add_argument("--all", action="store_true", help="Enable all three additional checks")
    a = p.parse_args()
    failures = 0
    passed = 0

    def run(label, task):
        nonlocal failures, passed
        try:
            msg = task()
            passed += 1
            print(f"[PASS] {label}: {msg}", flush=True)
        except Exception as error:
            failures += 1
            print(f"[FAIL] {label}: {type(error).__name__}: {error}", flush=True)

    print("Q3DGStream | GPU validation | no scene or viewer modifications", flush=True)
    run("GPU NTC forward", cuda_ntc_test)
    run("Real priority stages", stage_check)
    if a.check_quic or a.all:
        run("Local QUIC receiver", lambda: viewer_hooks(a.viewer))
    script = a.project / "research/evaluation/qntc_299_priority_eval.py"
    if a.verify_all or a.all:
        run("299-frame wire-byte parity",
            lambda: run_subprocess("wire parity",
                [sys.executable, str(script), "--verify-only",
                 "--out", "/tmp/q3dgstream_gpu_eval_113_115"], a.project))
    if a.evaluate_three or a.all:
        run("Three-frame CUDA saliency/random evaluation",
            lambda: run_subprocess("evaluation",
                [sys.executable, str(script), "--start", "113", "--end", "115",
                 "--seeds", "5", "--batch", "8192",
                 "--out", "/tmp/q3dgstream_gpu_eval_113_115"], a.project))
    print(f"\nRESULT: {passed} PASS, {failures} FAIL", flush=True)
    if failures == 0 and (a.all or (a.verify_all and a.evaluate_three and a.check_quic)):
        print("Offline GPU and local receiver checks passed. Live QUIC rendering still requires visual verification.", flush=True)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
