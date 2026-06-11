#!/usr/bin/env python3
"""
quantize_ntcs.py

Convert 3DGStream / 3DGS-QuaNTiC NTC files to FP16, INT8, or INT4.

Expected input layout:

    scene_root/
      init_3dgs.ply
      NTCs/
        config.json
        NTC_000000.pth
        NTC_000001.pth
        ...

Examples:

    python quantize_ntcs.py --src ~/qntc_scenes/flame_steak_official --out ~/qntc_scenes/flame_steak_fp16 --mode fp16 --overwrite --verify

    python quantize_ntcs.py --src ~/qntc_scenes/flame_steak_official --out ~/qntc_scenes/flame_steak_int8_b64 --mode int8 --block-size 64 --overwrite --verify

    python quantize_ntcs.py --src ~/qntc_scenes/flame_steak_official --out ~/qntc_scenes/flame_steak_int4_b64 --mode int4 --block-size 64 --overwrite --verify
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch


QTAG = "__3dgs_quantic_quantized_tensor__"


def load_pth(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out}. Use --overwrite.")
        shutil.rmtree(out)
    (out / "NTCs").mkdir(parents=True, exist_ok=True)


def copy_static_files(src: Path, out: Path) -> None:
    init_ply = src / "init_3dgs.ply"
    if init_ply.exists():
        shutil.copy2(init_ply, out / "init_3dgs.ply")
    else:
        print(f"[WARN] Missing {init_ply}")

    cfg = src / "NTCs" / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, out / "NTCs" / "config.json")
    else:
        print(f"[WARN] Missing {cfg}")


def quantize_fp16_tensor(t: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(t):
        return t.detach().cpu().half()
    return t.detach().cpu()


def quantize_int8_tensor(t: torch.Tensor, block_size: int) -> dict[str, Any]:
    shape = list(t.shape)
    x = t.detach().cpu().float().contiguous().view(-1)
    numel = x.numel()

    pad = (block_size - (numel % block_size)) % block_size
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=x.dtype)], dim=0)

    blocks = x.view(-1, block_size)
    max_abs = blocks.abs().amax(dim=1)
    scale = torch.clamp(max_abs / 127.0, min=1.0e-12)

    q = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)

    return {
        QTAG: True,
        "bits": 8,
        "method": "symmetric_block_int8",
        "shape": shape,
        "numel": int(numel),
        "block_size": int(block_size),
        "scale": scale.half(),
        "q": q.contiguous(),
    }


def pack_int4_signed(q: torch.Tensor) -> torch.Tensor:
    q = q.contiguous().view(-1).to(torch.int16)
    q = torch.clamp(q, -7, 7)

    # Store signed [-7, 7] as unsigned nibble [1, 15].
    u = (q + 8).to(torch.uint8)

    if u.numel() % 2:
        u = torch.cat([u, torch.full((1,), 8, dtype=torch.uint8)], dim=0)

    lo = u[0::2]
    hi = u[1::2] << 4
    return (lo | hi).contiguous()


def unpack_int4_signed(packed: torch.Tensor, padded_numel: int) -> torch.Tensor:
    packed = packed.cpu().contiguous().view(-1).to(torch.uint8)

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F

    u = torch.empty((packed.numel() * 2,), dtype=torch.uint8)
    u[0::2] = lo
    u[1::2] = hi
    u = u[:padded_numel]

    return u.to(torch.int16).sub(8).to(torch.float32)


def quantize_int4_tensor(t: torch.Tensor, block_size: int) -> dict[str, Any]:
    shape = list(t.shape)
    x = t.detach().cpu().float().contiguous().view(-1)
    numel = x.numel()

    pad = (block_size - (numel % block_size)) % block_size
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=x.dtype)], dim=0)

    blocks = x.view(-1, block_size)
    max_abs = blocks.abs().amax(dim=1)
    scale = torch.clamp(max_abs / 7.0, min=1.0e-12)

    q = torch.round(blocks / scale[:, None]).clamp(-7, 7).to(torch.int8)
    packed = pack_int4_signed(q)

    return {
        QTAG: True,
        "bits": 4,
        "method": "symmetric_block_int4_packed",
        "shape": shape,
        "numel": int(numel),
        "padded_numel": int(q.numel()),
        "block_size": int(block_size),
        "scale": scale.half(),
        "q_packed": packed,
    }


def quantize_obj(obj: Any, mode: str, block_size: int) -> Any:
    if isinstance(obj, torch.Tensor):
        if not torch.is_floating_point(obj):
            return obj.detach().cpu()

        if mode == "fp16":
            return quantize_fp16_tensor(obj)
        if mode == "int8":
            return quantize_int8_tensor(obj, block_size)
        if mode == "int4":
            return quantize_int4_tensor(obj, block_size)

        raise ValueError(f"Unsupported mode: {mode}")

    if isinstance(obj, dict):
        return {k: quantize_obj(v, mode, block_size) for k, v in obj.items()}

    if isinstance(obj, list):
        return [quantize_obj(v, mode, block_size) for v in obj]

    if isinstance(obj, tuple):
        return tuple(quantize_obj(v, mode, block_size) for v in obj)

    return obj


def dequantize_tensor(blob: dict[str, Any]) -> torch.Tensor:
    bits = int(blob["bits"])
    shape = tuple(blob["shape"])
    numel = int(blob["numel"])
    block_size = int(blob["block_size"])
    scale = blob["scale"].float().cpu()

    if bits == 8:
        q = blob["q"].float().cpu().view(-1, block_size)
        x = (q * scale[:, None]).contiguous().view(-1)[:numel]
    elif bits == 4:
        padded_numel = int(blob["padded_numel"])
        q = unpack_int4_signed(blob["q_packed"], padded_numel).view(-1, block_size)
        x = (q * scale[:, None]).contiguous().view(-1)[:numel]
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    return x.view(shape)


def dequantize_obj(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(QTAG, False):
        return dequantize_tensor(obj)

    if isinstance(obj, dict):
        return {k: dequantize_obj(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [dequantize_obj(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(dequantize_obj(v) for v in obj)

    return obj


def flatten_float_tensors(obj: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj):
            tensors.append(obj.detach().cpu().float().view(-1))
        return tensors

    if isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(flatten_float_tensors(v))
        return tensors

    if isinstance(obj, (list, tuple)):
        for v in obj:
            tensors.extend(flatten_float_tensors(v))
        return tensors

    return tensors


def verify_quantization(original: Any, quantized: Any, mode: str) -> dict[str, float]:
    reconstructed = quantized if mode == "fp16" else dequantize_obj(quantized)

    originals = flatten_float_tensors(original)
    reconstructions = flatten_float_tensors(reconstructed)

    if len(originals) != len(reconstructions):
        return {
            "tensor_count_match": 0.0,
            "rel_l2": float("nan"),
            "max_abs": float("nan"),
        }

    sq_err = 0.0
    sq_ref = 0.0
    max_abs = 0.0

    for a, b in zip(originals, reconstructions):
        n = min(a.numel(), b.numel())
        if n == 0:
            continue

        diff = a[:n] - b[:n].float()
        sq_err += float((diff * diff).sum().item())
        sq_ref += float((a[:n] * a[:n]).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))

    rel_l2 = math.sqrt(sq_err / max(sq_ref, 1.0e-12))

    return {
        "tensor_count_match": 1.0,
        "rel_l2": rel_l2,
        "max_abs": max_abs,
    }


def convert_scene(args: argparse.Namespace) -> None:
    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()

    ntc_dir = src / "NTCs"
    if not ntc_dir.exists():
        raise FileNotFoundError(f"Missing NTC directory: {ntc_dir}")

    ntc_files = sorted(ntc_dir.glob("NTC_*.pth"))
    if args.limit is not None:
        ntc_files = ntc_files[: args.limit]

    if not ntc_files:
        raise FileNotFoundError(f"No NTC_*.pth files found in {ntc_dir}")

    prepare_output(out, args.overwrite)
    copy_static_files(src, out)

    manifest: dict[str, Any] = {
        "mode": args.mode,
        "block_size": args.block_size if args.mode in {"int8", "int4"} else None,
        "ntc_count": len(ntc_files),
        "note": (
            "FP16 files store regular PyTorch tensors and are directly loadable by viewers "
            "expecting FP tensors. INT8/INT4 files store packed quantized blobs and require "
            "loader-side dequantization."
        ),
        "files": [],
    }

    total_in = 0.0
    total_out = 0.0

    for idx, ntc_path in enumerate(ntc_files, start=1):
        rel = ntc_path.relative_to(src)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        size_in = size_mb(ntc_path)
        obj = load_pth(ntc_path)
        qobj = quantize_obj(obj, args.mode, args.block_size)

        torch.save(qobj, dst)

        size_out = size_mb(dst)
        total_in += size_in
        total_out += size_out

        entry: dict[str, Any] = {
            "file": str(rel),
            "input_mb": size_in,
            "output_mb": size_out,
            "ratio": size_out / size_in if size_in else None,
        }

        if args.verify:
            entry.update(verify_quantization(obj, qobj, args.mode))

        manifest["files"].append(entry)

        print(
            f"[{idx:04d}/{len(ntc_files):04d}] {rel} "
            f"{size_in:.3f} MB -> {size_out:.3f} MB "
            f"({(size_out / size_in if size_in else 0):.3f}x)"
        )

    manifest["input_ntc_mb"] = total_in
    manifest["output_ntc_mb"] = total_out
    manifest["compression_ratio"] = total_out / total_in if total_in else None
    manifest["reduction_percent"] = (1.0 - total_out / total_in) * 100.0 if total_in else None

    manifest_path = out / "quantization_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("[DONE]")
    print(f"Source: {src}")
    print(f"Output: {out}")
    print(f"Mode: {args.mode}")
    print(f"NTCs: {len(ntc_files)}")
    print(f"Input total:  {total_in:.2f} MB")
    print(f"Output total: {total_out:.2f} MB")
    if total_in:
        print(f"Compression ratio: {total_out / total_in:.3f}x")
        print(f"Reduction: {(1.0 - total_out / total_in) * 100.0:.2f}%")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize 3DGS-QuaNTiC / 3DGStream NTC files to FP16, INT8, or INT4."
    )
    parser.add_argument("--src", required=True, type=Path, help="Input scene root.")
    parser.add_argument("--out", required=True, type=Path, help="Output scene root.")
    parser.add_argument("--mode", required=True, choices=["fp16", "int8", "int4"])
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N NTCs for testing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    convert_scene(args)


if __name__ == "__main__":
    main()
