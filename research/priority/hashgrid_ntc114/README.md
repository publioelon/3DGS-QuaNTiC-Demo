# Per-Gaussian NTC 114 HashGrid dependency mapping (read-only pilot)

This tool maps **which retained INT4 keep05 HashGrid parameter blocks are structurally read** when evaluating each persistent Gaussian from the initial Flame Steak scene. It uses the exact grid indexing algorithm of tiny-cuda-nn (`grid_scale`, level offset table, `pos_fract`, `grid_index`, CoherentPrime by default) and requires that randomly sampled Gaussian lookups **match CUDA gradients from the locally installed tiny-cuda-nn Encoding**. On any mismatch it fails before writing results.

## Run

```
cd ~/snap/firefox/common
unzip -o QNTC_Individual_Gaussian_Block_Map_NTC114_V1.zip
cd QNTC_Individual_Gaussian_Block_Map_NTC114_V1
bash run_pilot.sh --verify-only
bash run_pilot.sh
```

For a short smoke test before full mapping:

```
bash run_pilot.sh --max-gaussians 4096 --out "$HOME/snap/firefox/common/QNTC_individual_gaussian_smoke"
```

Full output goes to `~/snap/firefox/common/QNTC_individual_gaussian_ntc114`:

* `ntc_114_hash_block_to_gaussian_ids.npz`: compressed CSR (`block_ids`, `offsets`, `gaussian_ids`). For each index i, affected IDs are `gaussian_ids[offsets[i]:offsets[i+1]]`.
* `ntc_114_block_counts.csv`: counts per original retained HashGrid block ID.
* `ntc_114_report.json`: totals, top blocks and interpretation.

Query a block:

```
python query_block.py 128
```

## Rigorous interpretation

* **Exact structural indexing only after CUDA verification passes.** A block's inclusion means the tiny-cuda-nn HashGrid encoding reads its parameters at that input coordinate. Actual displacement may still be zero because of decoder weights, zero quantized parameters or cancellation.
* **Initial Gaussian XYZ**, not dynamically updated frame-114 positions. For per-frame structural access we must obtain the state immediately before applying NTC 114 and rerun with those coordinates.
* Only persistent Gaussian IDs are in this map; per-frame Gaussian additions have their own positions and aren't transformed by the persistent NTC in the existing renderer.
* MLP blocks have global potential effect and are returned separately; their exact numerical contribution must be probed separately.
* No viewer, original checkpoint or QUIC code is modified. Output files use modest storage (tens of MiB, subject to retained matches).
* If CUDA index verification fails, STOP. Do not disable the verification or use the output for scientific conclusions. The tiny-cuda-nn binary may use a different variant of the indexing algorithm.

Code references: tiny-cuda-nn `include/tiny-cuda-nn/encodings/grid.h`, `include/tiny-cuda-nn/common_device.h`.
