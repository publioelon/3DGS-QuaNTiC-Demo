"""Exact sparse-INT4 40-byte record transport; no quantization or visual recoloring.

Pilot protocol: 14-byte header !5sIBHH, followed by 40-byte records
  !If (original parameter block ID, float32 scale), then 32 packed bytes.
Each stage is one reliable application-prioritized QUIC file stream.
"""
import struct
import numpy as np

MAGIC = b'QNP01'
HEADER = struct.Struct('!5sIBHH')
RECORD = struct.Struct('!If32s')
BLOCK = 64
TOTAL = 31120
HASH = 30976


def encode_stage(frame, stage, records, expected_total=1556):
    if not 0 <= stage <= 3 or not 0 <= frame < 1_000_000:
        raise ValueError('bad stage/frame')
    if len(records) > expected_total:
        raise ValueError('too many records')
    out = bytearray(HEADER.pack(MAGIC, frame, stage, expected_total, len(records)))
    for idx, scale, packed in records:
        if not 0 <= idx < TOTAL or not np.isfinite(scale) or scale < 0 or len(packed) != 32:
            raise ValueError('invalid record')
        out.extend(RECORD.pack(int(idx), float(scale), packed))
    return bytes(out)


def decode_stage(data):
    if len(data) < HEADER.size:
        raise ValueError('truncated header')
    magic, frame, stage, total, count = HEADER.unpack_from(data)
    if magic != MAGIC or not 0 <= stage <= 3 or total != 1556 or count > 1556:
        raise ValueError('unexpected wire format')
    if len(data) != HEADER.size + count * RECORD.size:
        raise ValueError('invalid record count / byte length')
    records = [RECORD.unpack_from(data, HEADER.size + k * RECORD.size) for k in range(count)]
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate block within QUIC stage')
    return frame, stage, records


def decode_records_into(full, records):
    """Replace only records that have arrived, leaving all missing INT4 blocks zero."""
    if full.shape != (TOTAL, BLOCK) or full.dtype != np.float32:
        raise ValueError('expected (31120,64) float32 array')
    for idx, scale, packed in records:
        vals = np.frombuffer(packed, dtype=np.uint8)
        q = np.empty(BLOCK, dtype=np.float32)
        q[0::2] = (vals & 15).astype(np.float32) - 8
        q[1::2] = (vals >> 4).astype(np.float32) - 8
        full[idx] = q * float(scale)


def extract_records_from_torch(wire):
    p = 'model.params_sparse_int4_'
    idxs = wire[p+'block_indices'].view(-1).cpu().numpy()
    scales = wire[p+'scales'].view(-1).cpu().numpy()
    packed = wire['model.params_sparse_qint4_packed'].view(-1).cpu().numpy()
    if (str(wire[p+'quantization']) != 'offset_signed_int4_packed'
            or int(wire[p+'block_size']) != BLOCK
            or int(wire[p+'total_blocks']) != TOTAL
            or len(idxs) != 1556 or len(scales) != 1556 or len(packed) != 1556 * 32):
        raise ValueError('unexpected sparse INT4 source file')
    if len(set(map(int, idxs))) != len(idxs):
        raise ValueError('duplicate original IDs')
    return [(int(idx), float(scale), packed[i*32:(i+1)*32].tobytes())
            for i, (idx, scale) in enumerate(zip(idxs, scales))]
