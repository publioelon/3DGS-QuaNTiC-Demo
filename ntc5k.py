"""Compact post-training NTC codec with a hard per-update byte budget.

Format:
- protect final model.params blocks densely (int2 ternary or int4), intended for
  the small tiny-cuda-nn MLP tail;
- block-prune the preceding HashGrid region and transmit only highest-energy blocks;
- store one FP16 scale per transmitted block;
- preserve zero exactly;
- select sparse-block count from the actual byte budget.

This is a wire/storage codec. Runtime currently reconstructs dense FP32 model.params
before loading tiny-cuda-nn; sparse inference is a separate optimization.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch

MAGIC = b"N5K1"
VERSION = 1
_HEADER = struct.Struct("<4sBBBBHIHH6f")
_FLAG_TERNARY_INT2 = 1


def _packed_nbytes(block_size: int, bits: int) -> int:
    if bits not in (2, 4):
        raise ValueError("bits must be 2 or 4")
    return (block_size * bits + 7) // 8


def _pack_codes(codes: torch.Tensor, bits: int, block_size: int) -> bytes:
    vals = codes.detach().cpu().to(torch.uint8).view(-1)
    if vals.numel() != block_size:
        raise ValueError(f"expected {block_size} codes, got {vals.numel()}")

    if bits == 4:
        out = bytearray((block_size + 1) // 2)
        v = vals.tolist()
        for i in range(0, block_size, 2):
            lo = int(v[i]) & 0x0F
            hi = int(v[i + 1]) & 0x0F if i + 1 < block_size else 0
            out[i // 2] = lo | (hi << 4)
        return bytes(out)

    out = bytearray((block_size + 3) // 4)
    v = vals.tolist()
    for i in range(0, block_size, 4):
        byte = 0
        for j in range(4):
            if i + j < block_size:
                byte |= (int(v[i + j]) & 0x03) << (2 * j)
        out[i // 4] = byte
    return bytes(out)


def _unpack_codes(data: bytes, bits: int, block_size: int) -> torch.Tensor:
    if bits == 4:
        out = torch.empty(block_size, dtype=torch.uint8)
        for i in range(block_size):
            b = data[i // 2]
            out[i] = (b >> (4 * (i & 1))) & 0x0F
        return out

    if bits == 2:
        out = torch.empty(block_size, dtype=torch.uint8)
        for i in range(block_size):
            b = data[i // 4]
            out[i] = (b >> (2 * (i & 3))) & 0x03
        return out

    raise ValueError("bits must be 2 or 4")


def _quantize_block(x: torch.Tensor, bits: int) -> Tuple[float, torch.Tensor]:
    """Quantize one block while preserving exact zero as a reconstruction level."""
    x = x.detach().cpu().float().contiguous().view(-1)

    if bits == 4:
        max_abs = float(x.abs().max().item()) if x.numel() else 0.0
        scale = max(max_abs / 7.0, 1.0e-12)
        q = torch.round(x / scale).clamp(-7, 7).to(torch.int16)
        # Offset coding: signed [-7,7] -> [1,15], with zero -> 8.
        return scale, (q + 8).to(torch.uint8)

    if bits == 2:
        # Ternary PTQ: {-alpha, 0, +alpha}; fourth 2-bit symbol is reserved.
        abs_x = x.abs()
        mean_abs = float(abs_x.mean().item()) if x.numel() else 0.0
        threshold = 0.7 * mean_abs
        keep = abs_x > threshold
        scale = max(float(abs_x[keep].mean().item()), 1.0e-12) if bool(keep.any()) else 1.0e-12
        q = torch.zeros_like(x, dtype=torch.int8)
        q[(x > 0) & keep] = 1
        q[(x < 0) & keep] = -1
        codes = torch.zeros_like(q, dtype=torch.uint8)
        codes[q > 0] = 1
        codes[q < 0] = 2
        return scale, codes

    raise ValueError("bits must be 2 or 4")


def _dequantize_block(scale: float, codes: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 4:
        return (codes.to(torch.int16) - 8).float() * float(scale)
    if bits == 2:
        q = torch.zeros(codes.numel(), dtype=torch.float32)
        q[codes == 1] = 1.0
        q[codes == 2] = -1.0
        return q * float(scale)
    raise ValueError("bits must be 2 or 4")


def budget_plan(
    numel: int,
    budget_bytes: int = 5 * 1024,
    block_size: int = 64,
    hash_bits: int = 2,
    tail_bits: int = 2,
    tail_params: int = 9216,
) -> Dict[str, int]:
    if numel <= 0:
        raise ValueError("numel must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if hash_bits not in (2, 4) or tail_bits not in (2, 4):
        raise ValueError("hash_bits/tail_bits must be 2 or 4")

    total_blocks = math.ceil(numel / block_size)
    if total_blocks > 65535:
        raise ValueError(
            f"{total_blocks} blocks exceed uint16 index capacity; increase --block-size"
        )

    tail_blocks = min(total_blocks, math.ceil(max(0, tail_params) / block_size))
    hash_blocks_total = total_blocks - tail_blocks
    tail_record = 2 + _packed_nbytes(block_size, tail_bits)
    hash_record = 4 + _packed_nbytes(block_size, hash_bits)
    fixed_bytes = _HEADER.size + tail_blocks * tail_record
    remaining = budget_bytes - fixed_bytes
    if remaining < 0:
        raise ValueError(
            f"Protected tail alone needs {fixed_bytes} bytes, above budget {budget_bytes}. "
            "Reduce --tail-params, --block-size, or --tail-bits."
        )

    selected_hash_blocks = min(hash_blocks_total, remaining // hash_record)
    encoded_bytes = fixed_bytes + selected_hash_blocks * hash_record
    return {
        "budget_bytes": int(budget_bytes),
        "header_bytes": int(_HEADER.size),
        "numel": int(numel),
        "block_size": int(block_size),
        "total_blocks": int(total_blocks),
        "tail_blocks": int(tail_blocks),
        "tail_effective_params": int(min(numel, tail_blocks * block_size)),
        "hash_blocks_total": int(hash_blocks_total),
        "selected_hash_blocks": int(selected_hash_blocks),
        "hash_record_bytes": int(hash_record),
        "tail_record_bytes": int(tail_record),
        "encoded_bytes": int(encoded_bytes),
        "slack_bytes": int(budget_bytes - encoded_bytes),
    }


def encode_ntc5k(
    params: torch.Tensor,
    xyz_min: torch.Tensor,
    xyz_max: torch.Tensor,
    *,
    budget_bytes: int = 5 * 1024,
    block_size: int = 64,
    hash_bits: int = 2,
    tail_bits: int = 2,
    tail_params: int = 9216,
    score_mode: str = "l2",
) -> Tuple[bytes, Dict[str, float]]:
    x = params.detach().cpu().float().contiguous().view(-1)
    plan = budget_plan(x.numel(), budget_bytes, block_size, hash_bits, tail_bits, tail_params)

    total_blocks = plan["total_blocks"]
    padded = torch.zeros(total_blocks * block_size, dtype=torch.float32)
    padded[: x.numel()] = x
    blocks = padded.view(total_blocks, block_size)

    n_tail = plan["tail_blocks"]
    n_hash_total = plan["hash_blocks_total"]
    n_hash_keep = plan["selected_hash_blocks"]

    if n_hash_keep:
        hb = blocks[:n_hash_total]
        if score_mode == "l2":
            scores = (hb * hb).sum(dim=1)
        elif score_mode == "l1":
            scores = hb.abs().sum(dim=1)
        elif score_mode == "max":
            scores = hb.abs().amax(dim=1)
        else:
            raise ValueError("score_mode must be one of: l2, l1, max")
        selected = torch.topk(scores, k=n_hash_keep, largest=True, sorted=False).indices
        selected, _ = torch.sort(selected)
    else:
        selected = torch.empty(0, dtype=torch.long)

    bmin = xyz_min.detach().cpu().float().view(-1)
    bmax = xyz_max.detach().cpu().float().view(-1)
    if bmin.numel() != 3 or bmax.numel() != 3:
        raise ValueError("xyz_min and xyz_max must contain three values each")

    out = bytearray(_HEADER.pack(
        MAGIC, VERSION, hash_bits, tail_bits, _FLAG_TERNARY_INT2,
        block_size, x.numel(), int(selected.numel()), n_tail,
        float(bmin[0]), float(bmin[1]), float(bmin[2]),
        float(bmax[0]), float(bmax[1]), float(bmax[2]),
    ))

    for idx in selected.tolist():
        scale, codes = _quantize_block(blocks[idx], hash_bits)
        out += struct.pack("<H", int(idx))
        out += struct.pack("<e", float(scale))
        out += _pack_codes(codes, hash_bits, block_size)

    # Tail block indices are implicit/contiguous: no index overhead.
    for idx in range(total_blocks - n_tail, total_blocks):
        scale, codes = _quantize_block(blocks[idx], tail_bits)
        out += struct.pack("<e", float(scale))
        out += _pack_codes(codes, tail_bits, block_size)

    if len(out) > budget_bytes:
        raise AssertionError(f"codec bug: produced {len(out)} > budget {budget_bytes}")

    retained = min(x.numel(), n_tail * block_size + int(selected.numel()) * block_size)
    stats: Dict[str, float] = {
        **{k: float(v) for k, v in plan.items()},
        "actual_bytes": float(len(out)),
        "retained_parameter_slots": float(retained),
        "retained_fraction": float(retained / x.numel()),
        "hash_bits": float(hash_bits),
        "tail_bits": float(tail_bits),
    }
    return bytes(out), stats


def decode_ntc5k_bytes(blob: bytes) -> Dict[str, torch.Tensor]:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated NTC5K file")

    fields = _HEADER.unpack_from(blob, 0)
    magic, version, hash_bits, tail_bits, flags, block_size, numel, n_hash, n_tail, *bounds = fields
    if magic != MAGIC:
        raise ValueError(f"invalid NTC5K magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported NTC5K version: {version}")
    if hash_bits not in (2, 4) or tail_bits not in (2, 4):
        raise ValueError("corrupt NTC5K bit width")

    total_blocks = math.ceil(numel / block_size)
    if n_tail > total_blocks:
        raise ValueError("corrupt NTC5K tail block count")

    padded = torch.zeros(total_blocks * block_size, dtype=torch.float32)
    offset = _HEADER.size

    hash_payload = _packed_nbytes(block_size, hash_bits)
    for _ in range(n_hash):
        if offset + 4 + hash_payload > len(blob):
            raise ValueError("truncated NTC5K hash record")
        (idx,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        (scale,) = struct.unpack_from("<e", blob, offset)
        offset += 2
        data = blob[offset : offset + hash_payload]
        offset += hash_payload
        if idx >= total_blocks - n_tail:
            raise ValueError("corrupt NTC5K sparse block index")
        codes = _unpack_codes(data, hash_bits, block_size)
        padded[idx * block_size : (idx + 1) * block_size] = _dequantize_block(
            scale, codes, hash_bits
        )

    tail_payload = _packed_nbytes(block_size, tail_bits)
    for idx in range(total_blocks - n_tail, total_blocks):
        if offset + 2 + tail_payload > len(blob):
            raise ValueError("truncated NTC5K tail record")
        (scale,) = struct.unpack_from("<e", blob, offset)
        offset += 2
        data = blob[offset : offset + tail_payload]
        offset += tail_payload
        codes = _unpack_codes(data, tail_bits, block_size)
        padded[idx * block_size : (idx + 1) * block_size] = _dequantize_block(
            scale, codes, tail_bits
        )

    if offset != len(blob):
        raise ValueError(f"unexpected trailing bytes in NTC5K: {len(blob) - offset}")

    return {
        "model.params": padded[:numel].contiguous(),
        "xyz_bound_min": torch.tensor(bounds[:3], dtype=torch.float32),
        "xyz_bound_max": torch.tensor(bounds[3:], dtype=torch.float32),
        "_ntc5k_flags": torch.tensor([flags], dtype=torch.int32),
    }


def load_ntc5k(path: str | Path) -> Dict[str, torch.Tensor]:
    return decode_ntc5k_bytes(Path(path).read_bytes())
