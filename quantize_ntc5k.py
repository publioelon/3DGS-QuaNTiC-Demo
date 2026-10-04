#!/usr/bin/env python3
"""Post-training network-aware sparsity + quantization for 3DGStream NTCs.

Produces custom NTC_XXXXXX.ntc5k files under a hard byte budget (default 5 KiB).
The codec protects a contiguous tail of tiny-cuda-nn model.params (intended
for the small MLP) and block-prunes the preceding HashGrid parameter region.

Example:
    python quantize_ntc5k.py \
      --src ~/qntc_scenes/flame_steak_official \
      --out ~/qntc_scenes/flame_steak_ntc5k \
      --budget-bytes 5120 --block-size 64 \
      --hash-bits 2 --tail-bits 2 --tail-params 9216 \
      --verify --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

from ntc5k import encode_ntc5k, load_ntc5k


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state(obj: Any) -> Dict[str, Any]:
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dict")
    return state


def _get_tensor(state: Dict[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"checkpoint missing tensor {key!r}")
    return value


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().float().view(-1)
    b = b.detach().cpu().float().view(-1)
    d = a - b
    return math.sqrt(float((d * d).sum().item()) / max(float((a * a).sum().item()), 1e-12))


def _copy_scene_scaffolding(src: Path, out: Path, copy_additions: bool) -> None:
    (out / "NTCs").mkdir(parents=True, exist_ok=True)
    init_ply = src / "init_3dgs.ply"
    if init_ply.exists():
        shutil.copy2(init_ply, out / "init_3dgs.ply")

    cfg = src / "NTCs" / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"missing NTC config: {cfg}")
    shutil.copy2(cfg, out / "NTCs" / "config.json")

    additions = src / "additional_3dgs"
    if copy_additions and additions.is_dir():
        shutil.copytree(additions, out / "additional_3dgs", dirs_exist_ok=True)


def convert(args: argparse.Namespace) -> None:
    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    ntc_dir = src / "NTCs"
    files = sorted(ntc_dir.glob("NTC_*.pth"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no NTC_*.pth found in {ntc_dir}")

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {out}; use --overwrite")
        shutil.rmtree(out)
    _copy_scene_scaffolding(src, out, args.copy_additions)

    manifest: Dict[str, Any] = {
        "format": "NTC5K-v1",
        "source": str(src),
        "budget_bytes": args.budget_bytes,
        "block_size": args.block_size,
        "hash_bits": args.hash_bits,
        "tail_bits": args.tail_bits,
        "tail_params_requested": args.tail_params,
        "score_mode": args.score_mode,
        "files": [],
    }

    for i, path in enumerate(files, 1):
        ckpt = _torch_load(path)
        state = _extract_state(ckpt)
        params = _get_tensor(state, "model.params")
        xyz_min = _get_tensor(state, "xyz_bound_min")
        xyz_max = _get_tensor(state, "xyz_bound_max")

        blob, stats = encode_ntc5k(
            params, xyz_min, xyz_max,
            budget_bytes=args.budget_bytes,
            block_size=args.block_size,
            hash_bits=args.hash_bits,
            tail_bits=args.tail_bits,
            tail_params=args.tail_params,
            score_mode=args.score_mode,
        )

        dst = out / "NTCs" / (path.stem + ".ntc5k")
        dst.write_bytes(blob)
        actual = dst.stat().st_size
        if actual > args.budget_bytes:
            raise AssertionError(f"{dst} is {actual} bytes > budget {args.budget_bytes}")

        entry: Dict[str, Any] = {
            "source_file": path.name,
            "output_file": dst.name,
            **stats,
        }
        if args.verify:
            recon = load_ntc5k(dst)
            entry["rel_l2_model_params"] = _rel_l2(params, recon["model.params"])
            entry["bounds_max_abs_error"] = max(
                float((xyz_min.float().view(-1) - recon["xyz_bound_min"]).abs().max().item()),
                float((xyz_max.float().view(-1) - recon["xyz_bound_max"]).abs().max().item()),
            )

        manifest["files"].append(entry)
        keep_pct = 100.0 * float(stats["retained_fraction"])
        verify_msg = f" relL2={entry['rel_l2_model_params']:.6f}" if args.verify else ""
        print(
            f"[{i:04d}/{len(files):04d}] {path.name} -> {dst.name}: "
            f"{actual} B ({actual/1024:.3f} KiB), retained_slots={keep_pct:.3f}%{verify_msg}"
        )

    manifest_path = out / "ntc5k_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] {len(files)} NTCs -> {out}")
    print(f"Hard budget per NTC: {args.budget_bytes} bytes ({args.budget_bytes/1024:.3f} KiB)")
    print(f"Manifest: {manifest_path}")
    if not args.copy_additions:
        print("[NOTE] additional_3dgs was not copied. Use --copy-additions for standalone playback.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hard-budget NTC5K post-training compressor")
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--budget-bytes", type=int, default=5 * 1024)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--hash-bits", type=int, choices=[2, 4], default=2)
    p.add_argument("--tail-bits", type=int, choices=[2, 4], default=2)
    p.add_argument(
        "--tail-params", type=int, default=9216,
        help=("Requested number of final model.params coefficients to protect densely. "
              "For the current 64-wide 2-hidden-layer NTC, 9216 is the initial "
              "architecture-aware hypothesis; rounded to whole blocks.")
    )
    p.add_argument("--score-mode", choices=["l2", "l1", "max"], default="l2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--copy-additions", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget_bytes <= 0 or args.block_size <= 0 or args.tail_params < 0:
        raise ValueError("budget/block size must be positive and tail params non-negative")
    convert(args)


if __name__ == "__main__":
    main()
