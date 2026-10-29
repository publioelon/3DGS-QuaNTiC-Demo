# QNP01: sparse-INT4 NTC priority stage format

**`.qblk` is a project-defined binary file extension**, not a standard PyTorch or QUIC model type. QUIC sends its bytes, and our decoder interprets them.

Every stage contains a **14-byte** network-order header, followed by zero or more **40-byte** weight records.

```python
import struct
HEADER = struct.Struct('!5sIBHH')  # 14 bytes
RECORD = struct.Struct('!If32s')  # 40 bytes
```

| Header field | Width | Meaning |
|---|---:|---|
| magic | 5 B | ASCII `QNP01` |
| NTC frame index | 4 B | Sequential animation-step ID |
| priority index | 1 B | P0, P1, P2 or P3 |
| total block count | 2 B | 1,556 for current INT4 keep05 |
| stage block count | 2 B | Number of records in this stage |

| Record field | Width | Meaning |
|---|---:|---|
| Block index | 4 B | Index in the original 31,120-block parameter array |
| FP32 scale | 4 B | One scale per retained block |
| Packed INT4 | 32 B | 64 4-bit weights |

The low and high nibbles encode signed integers through a +8 offset: `q = nibble - 8`, representing -8 through 7. Reconstruct floating-point weight as `w = scale * q`. The block's ID determines its insertion position; all other weights remain zero until their stage arrives.

For NTC 114 the stages contained 396, 387, 386 and 387 records. Reassembling all four stages reconstructed the source INT4 block IDs, scales and packed values byte-for-byte.

**Do not confuse** the custom `.qblk` stage files with the original `.pth` checkpoint, `.npz` priority ordering, or `.ply` Gaussian geometry.
