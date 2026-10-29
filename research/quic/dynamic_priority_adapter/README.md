# QNTC Dynamic 299 schedules → existing progressive QUIC receiver (pilot V1)

This is a **standalone sender**. It does **not** modify any file in `~/3DGS-QuaNTiC-Demo`, the FP32 source, the INT4 source, or the 299 validated dynamic schedules. It does not re-rank blocks using NTC114. It sends byte-for-byte the generated prevalidated `.qblk` stage files. It also sends the 299 original additions files under the exact filenames that the existing `LiveTCPState` / `LiveQUICState` recognizes.

**Scope:** Actual QUIC file delivery with an application-defined priority order. This is NOT QUIC transport-native stream prioritization. The source is still INT4 **keep05**: full reception of these stages does NOT reproduce FP32 visual fidelity. Source compression itself already removes 95% of blocks. Stage transmission can demonstrate *progressive quality within the compressed source*.

## Inputs (already on the reference GPU workstation)

- `~/qntc_scenes/flame_steak_official/init_3dgs.ply`, and `additional_3dgs/additions_000000.ply` ... `additions_000298.ply`.
- `/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat/scenes/candidate/{init_3dgs.ply,NTCs/config.json}`.
- Same experiment's `wire/NTC_000000.pth` for original checkpoint bounds.
- `~/snap/firefox/common/QNTC_dynamic_299_priorities_v1/stages/NTC_000000_P0.qblk` through `NTC_000298_P3.qblk`.

No need to create a new scene or symlink additions. In particular, the accidental recursive `additional_3dgs/additional_3dgs` link is ignored.

## Preparation

Close any other viewer/QUIC sender listening on UDP 5002. Extract this package in `~/snap/firefox/common`.

```bash
cd ~/snap/firefox/common/QNTC_Dynamic_299_QUIC_Integration_V1
bash preflight.sh
```

Preflight checks matching initial PLY hashes, all 30 additions and 120 stages for the first 30 updates, unique block IDs and four-stage coverage (1556 original INT4 blocks per NTC), and sufficient free disk space. It doesn't edit or transmit anything.

If the working renderer has not been backed up yet, preserve it manually (`cp -n`), without overwriting earlier verified backups:

```bash
cd ~/3DGS-QuaNTiC-Demo
cp -n renderer_cuda.py renderer_cuda.py.WORKING_BACKUP
```

## Test 30 updates (not the complete animation)

Terminal 1:

```bash
cd ~/snap/firefox/common/QNTC_Dynamic_299_QUIC_Integration_V1
bash run_receiver.sh
```

`run_receiver.sh` clears **only** `/tmp/qntc_dynamic_priority_cache`, not the validated `/tmp/qntc_fp32_reference_cache`. The receiver runs with priority overlay disabled, original SH colors, scale 1.0, rotation `current`, fixed camera, 30 display FPS target and playback looping.

Terminal 2:

```bash
cd ~/snap/firefox/common/QNTC_Dynamic_299_QUIC_Integration_V1
bash run_sender.sh
```

The pilot uses `--start 0 --end 29` by default so it checks 30 updates, with `QNTC_STAGE_DELAY=0` by default to avoid intentional slowing. It streams additions before each frame's 4 priority stages. It may display only a short segment, since only 30 of 299 NTCs arrive. This is an initial functional test, not a full-sequence fidelity or network performance result.

Check logs:

```bash
grep -E 'QUIC CPU READY|REAL QUIC ERROR|draw error|additions_' /tmp/qntc_dynamic_priority_receiver.log | tail -60
```

## Full sequence after pilot

Close both processes, restart receiver (clears **only** the dedicated dynamic cache), then:

```bash
QNTC_END=298 bash run_sender.sh
```

Use a different stage delay for visible progressive intermediate states, e.g. `QNTC_STAGE_DELAY=0.02 bash run_sender.sh`. Note four `0.02`-second sleeps per NTC fundamentally limit motion update rate to at most 12.5 updates/s excluding QUIC, render and decoding costs; do not claim 30 NTC/s in that case.

## Known issues / limitations

- Receiver `qntc_progressive_live.py` is the installed bounded v2 sidecar. This package does **not** change the receiver. Runtime behavior with actual additions has not been GPU tested in the model environment.
- The existing `cat_additions()` assumes each requested addition is available. Sender commits addition before corresponding NTC stages, but if a race still occurs, inspect receiver logs first; do not edit the working renderer.
- Existing progressive renderer behavior may accumulate errors across partially applied NTCs; compare against the verified FP32 local reference and the *complete INT4 compressed baseline*. Never claim this adapter fixes cumulative drift.
- QUIC reliability and exact file bytes are addressed here. Video reconstruction quality, FPS, latency, and saliency benefit remain to be evaluated empirically.
