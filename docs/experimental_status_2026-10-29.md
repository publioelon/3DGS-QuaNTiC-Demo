# Experiment status · 29 October 2026

## Verified

- **299 of 299** four-stage `.qblk` sets match their original compressed `int4_keep05` wire checkpoints at the retained block level.
- Student trajectory: **300 states** replayed. All **14** reference checkpoint states checked exactly for XYZ and stored quaternion component values.
- Priority ordering vs random: **299 frames**, **five random seeds**, **four cumulative block budgets**, separated by saliency group. At approximately 75% block reception, the prior analysis observed around **77%–78% lower relative RMS translation reconstruction error** for red and orange groups compared with mean random ordering.
- NTC 114 HashGrid-to-Gaussian dependency mapping passed **12 of 12** sampled tiny-cuda-nn gradient checks.
- Experimental **30-update local QUIC pilot** received 120 priority stages and 30 additional Gaussian files.

## Not yet fully validated in a unified GitHub checkout

The GitHub `main` source predates recent QUIC hooks. Its older TCP receiver is not interchangeable with the locally modified QUIC viewer. Before advertising a one-command 299-frame real-time demonstration, sync and test the actual local versions of `main.py`, `renderer_cuda.py`, `qntc_progressive_live.py` and relevant QUIC transport code.

The live pilot's send ordering is **application-level**, not transport-native QUIC stream scheduling. Stage delay introduced to display intermediate states lowers actual NTC update throughput. The offline 299-frame comparison studies **isolated frame transforms**, not accumulated live playback quality.

Rotation v1 excluded some zero-norm quaternions; rotation v2 tests an explicit identity fallback that has not been confirmed to match the renderer exactly.

## Benchmark protocol integrity

The earlier Flame Steak PQAT reference **33.938 dB / SSIM 0.9591 / LPIPS 0.0776** came from an earlier **50-frame** evaluation. Later lowbit299 compression quality values use a different rendering evaluation protocol over **299** frames, with a different baseline. Do **not** rank image quality across these protocols without reevaluating using one consistent method.

## Assets not in Git

The 299 source `.pth` checkpoints, sparse `.qblk` output stages, compressed student states, SalFoM maps and large scene `.ply` files are generated/model data and are not stored in source control.
