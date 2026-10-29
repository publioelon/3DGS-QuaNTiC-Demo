# QNTC Dynamic Saliency-Aware NTC Block Priorities V1

This is **offline preparation** for all 299 NTCs, NOT yet a validated real-time QUIC quality result. Nothing edits `~/3DGS-QuaNTiC-Demo/main.py`, `renderer_cuda.py`, your original FP32 scene, compressed checkpoint files, or the existing sender/receiver. You must run it in the working `qntcstream` virtualenv on the Ubuntu GPU machine.

## What it does

1. Builds a 32-tile importance timeline from the existing **36 saved 12-camera SalFoM 3D anchors** using `prepare_dynamic_saliency.py`. The timeline uses measured anchors and interpolates scores between them; it does not claim new SalFoM predictions for every intermediate frame.
2. For each selected sparse INT4 keep05 PQAT NTC, loads its own 1,556 retained 64-parameter block records, reconstructs its **exact compressed** weights, and re-measures sensitivity with tiny-cuda-nn gradients for sampled Gaussians in each of 32 persistent spatial groups. Unlike the earlier pilot, this does **not** reuse NTC114 gradient support for other checkpoints.
3. Combines per-NTC block sensitivities with that frame's saliency tile weights. Assigns two top-ranked tiles P0, two P1, four nearby P2, other P3. Shared HashGrid blocks are scored across all tiles they influence. The decoder MLP blocks go first, then HashGrid records in descending importance.
4. Saves an `orders/NTC_XXXXXX.npz` schedule for each frame. With `--emit-stages`, saves four `stages/NTC_XXXXXX_P[0-3].qblk` blobs of original 40-byte INT4 records, ready for a later QUIC sender integration. The `.qblk` format is unchanged. `check_schedule.py --stages` verifies that all staged binary records equal their source checkpoint records, bit-for-bit.

## Commands

Install package in your downloads folder, then:

```bash
cd ~/snap/firefox/common
unzip -o QNTC_Dynamic_299_Priorities_V1.zip -d QNTC_Dynamic_299_Priorities_V1
cd QNTC_Dynamic_299_Priorities_V1
bash run_pilot.sh
```

Pilot checks CPU-only synthetic tests; generates the saved-saliency timeline; and ranks NTC114 with 32 tiles, 128 sampled Gaussians per tile, and 2 gradient probes per tile. It verifies the stage bytes match the original source. Outputs are saved under `~/snap/firefox/common/QNTC_dynamic_299_priorities_v1`.

Once NTC114 passes, check whether schedules change over a small sequence:

```bash
bash run_small_sequence.sh
```

Then extend to all checkpoints:

```bash
bash run_all.sh
```

**Full run is computationally heavy** (up to 299×32×2 gradient probes). Each NTC is processed sequentially, with no full sequence loaded in GPU memory; the command skips completed outputs to support restart. Do not run with other heavy GPU workloads. It creates ~20 MB of stage payloads and a few MB of compact schedules (not another 2.3 GiB scene copy).

Check just a range without retransmitting anything:

```bash
python rank_ntc_frames.py --start 110 --end 118 --emit-stages
python check_schedule.py --start 110 --end 118 --stages
```

The full run uses the existing INT4 keep05 **PQAT** wire files. A high-quality compressor is still needed: a correct priority order does not recover the 95% of blocks absent from this sparse format. The successful buffered FP32 QUIC baseline is preserved.

## Technical limitations and interpretation

- Sensitivity is approximate (`(d output / d weight × weight)^2` over randomized output probes). The input XYZ sample comes from **fixed initial persistent Gaussian coordinates**, not exact deformed Gaussian coordinates at every frame. This is a time-varying *per-NTC* sensitivity estimate, not a full motion-trajectory derivative. To use actual frame-state positions requires validating Gaussian state snapshots first.
- Saliency comes from saved multiview **fused 3D SalFoM** anchors. The rank based P0/P1 groups are **not** verified per-camera gaze consensus or exact visibility. 36 anchor frames, not 299 new inference runs.
- Only 280,768 persistent Gaussians are spatially represented. Per-frame newly added Gaussians have no persistent ID association in this experiment.
- Not yet live QUIC streaming, no image PSNR or QoE results. Do **not** report it as such. Implement the sender's schedule support after pilot checks.
- All output directories are in the user's home snap location; source files and the working viewer remain untouched. Never invoke `run_receiver.sh` with `--quic_clear_cache` when preserving your verified FP32 cache.
