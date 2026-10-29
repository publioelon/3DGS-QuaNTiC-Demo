# QNTC real QUIC progressive receiver: 299-frame repair (v2)

This fixes the short repeated 20-frame experiment, stale/evicted CPU NTC states,
and heavy whole-network CPU reconstruction on the UI thread. It does **not**
change the current Gaussian colors, alter INT4 source checkpoints, retrain models,
or change the existing CUDA renderer or `main.py`.

## Mechanism

- Retains the reliable, actual QUIC file-stream transport from the existing pilot.
- Sends NTC indices **0 through 298**, as four real staged QUIC transmissions.
- Receives compact `.qblk` chunks on disk (around 19 MiB for the full 299-update
  sequence). The CPU worker decodes only the current/nearby NTC states, retaining
  at most 10 dense CPU states, rather than ~2.4 GiB for all 299 dense states.
- Vectorized sparse INT4 unpacking avoids ~1556 Python record loops per decode.
- Publishes a new NTC stage version only *after* CPU preparation finishes.
- The installed renderer's existing progressive-redraw hook then re-applies
  newer stages to the same visible NTC step. The previous NTC is required to
  finish before proceeding to the next one; no missing frames are substituted.
- The frame renderer remains on its existing main/CUDA thread. CUDA inference
  and GPU parameter uploads are still synchronous and may limit actual FPS.

## Installation (close viewer and sender first)

Download `QNTC_Progressive_299_Fix.zip` to `~/snap/firefox/common`.

```bash
cd ~/snap/firefox/common
unzip -o QNTC_Progressive_299_Fix.zip
source ~/venvs/qntcstream/bin/activate
cd QNTC_Progressive_299_Fix
python test_quic_v2.py
python test_receiver_state_machine_v2.py
python install_bounded_progressive_v2.py --viewer ~/3DGS-QuaNTiC-Demo
# only if [DRY RUN OK]:
python install_bounded_progressive_v2.py --viewer ~/3DGS-QuaNTiC-Demo --apply
```

The installer first verifies that the prior pilot's `main.py` and
`renderer_cuda.py` hooks exist. It changes ONLY `qntc_progressive_live.py`
and `qntc_progressive_send.py`, with per-file backups named
`.before_bounded_progressive_v2`.

## Run in two terminals

Terminal 1 (viewer), leave open:

```bash
cd ~/snap/firefox/common/QNTC_Progressive_299_Fix
bash run_receiver.sh
```

If local QUIC certificate expired or missing, recreate once:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 7 \
  -subj '/CN=localhost' \
  -keyout /tmp/qntc_progressive.key \
  -out /tmp/qntc_progressive.crt
```

Terminal 2 (sender), once `[QUIC] listening ...` appears in Terminal 1:

```bash
cd ~/snap/firefox/common/QNTC_Progressive_299_Fix
bash run_sender.sh
```

The receiver disables colored overlays and looping. Its playback target is
30 display FPS, but actual FPS depends on GPU inference and drawing workload.

### Stage spacing is for visual inspection, not a 30 NTC/s benchmark

Default delay: `0.045` seconds after each of four priority chunks, allowing
several normal-looking render frames to show intermediate NTC states. This
intentionally means **about 5.5 NTC motion updates/s**, even if the window
continues rendering at 30 FPS. For faster full animation:

```bash
QNTC_STAGE_DELAY=0.008 bash run_sender.sh
```

At very short stage spacing, chunks may all arrive before the next displayed
frame; the viewer will then show full NTCs and little visible progression.

### Logs

```bash
grep -E 'QUIC CPU READY|REAL QUIC ERROR|draw error' /tmp/qntc_receiver_v2.log | tail -45
tail -20 /tmp/qntc_sender_v2.log
```

*Receiver* logs `[QUIC CPU READY] NTC=... stages=1/4 ... 4/4` for processed
updates. Full 299 updates cover approximately 10 seconds of source animation at
30 NTC updates/s, but the staged demonstration deliberately plays motion
more slowly. Do **not** set `LOOP_PLAYBACK=1` for this experiment.

## Accuracy limits

- This is real QUIC application-ordered delivery, but not transport-internal
  QUIC stream-priority control or a congestion-controlled real 5G experiment.
- The sender uses NTC114 **binary** HashGrid support plus interpolated 3D SalFoM
  priority weights. Unknown parameter blocks rank last. It does **not**
  reproduce the earlier exact weighted-Jacobian ordering or 81% P0 RMSE result.
- Shared decoder parameters are included in P0 if present in each NTC. Some
  sparse frames have none; we do not invent or repair missing MLP blocks.
- Stage P0..P3 are successive delivery quartiles, **not** independent
  geometric tiles or validated multi-camera-overlap priority classes.
- This code was tested with local CPU-only synthetic stages and against the
  prior viewer-source hooks, not actual RTX4060 CUDA playback.
- If motion remains small even after the full NTC, compare offline full INT4
  playback to isolate source quantization and renderer translation scaling.

## Undo

Close viewer and sender, then:

```bash
cd ~/snap/firefox/common/QNTC_Progressive_299_Fix
python install_bounded_progressive_v2.py --viewer ~/3DGS-QuaNTiC-Demo --restore --apply
```

This restores only the experimental sidecars, preserving your prior main/renderer
backup files and the original scene.
