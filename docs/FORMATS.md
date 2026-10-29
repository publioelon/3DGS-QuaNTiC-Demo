# Data and wire formats

| Extension | Role | Typical location |
|---|---|---|
| `.pth` | PyTorch NTC checkpoint: FP32 original, quantized wire or decoded runtime | scene `NTCs/`, `wire/`, `runtime/` |
| `.qblk` | Custom priority-stage binary records for sparse INT4 weights | `QNTC_dynamic_299_priorities_v1/stages/` |
| `.npz` | NumPy priority schedules or structural Gaussian-to-block lookup | `orders/`, mapping outputs |
| `.ply` | Initial 3DGS and frame-specific Gaussian additions | scene root and `additional_3dgs/` |
| `.pt` | Saved Gaussian states for offline evaluation; not normally streamed | `QNTC_299_student_states/` |
| `.json` | Config, saliency timeline, verification reports | various |

## `.qblk` (our own format)

**Header:** network byte order `!5sIBHH` (14 bytes).

| Field | Bytes | Meaning |
|---|---:|---|
| magic | 5 | `QNP01` |
| frame index | 4 | the specific NTC step |
| stage index | 1 | P0–P3 |
| total retained blocks | 2 | e.g. 1556 |
| blocks in stage | 2 | variable |

**Each record:** `!If32s` (40 bytes): block index (4), FP32 scale (4), packed INT4 weights (32 = 64 × 4 bits).

For every two weights one byte holds two unsigned nibbles. Our encoding uses **offset signed INT4**: `q = nibble - 8`, so the 16 allowed integers are `−8…7`. Dequantize each entry as `w ≈ scale × q`.

A `.qblk` file is not executable. Its extension is arbitrary; it only works with our encoder/decoder. A complete NTC is reconstructed by inserting each received 64-weight block into its proper original parameter positions; other blocks are initially zero.

**Caution:** Four independent `qblk` stage files do not constitute four independent Gaussian tiles. HashGrid blocks can influence many Gaussians. Always decode into the correct NTC frame and verify all unique record indices.
