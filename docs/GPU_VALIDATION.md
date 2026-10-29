# GPU validation

Use the Ubuntu machine with an NVIDIA GPU, the saved Flame Steak assets and the `qntcstream` environment. The public GitHub viewer still lacks the latest local QUIC integration. This procedure **never overwrites the working viewer**.

## Run the reproducible GPU tests

Clone the repository into a **separate directory** to avoid mixing older public code with locally modified renderer files:

```bash
cd ~
git clone https://github.com/publioelon/3DGS-QuaNTiC-Demo.git Q3DGStream-validation
source ~/venvs/qntcstream/bin/activate
cd ~/Q3DGStream-validation
python scripts/gpu_validation.py --all --viewer ~/3DGS-QuaNTiC-Demo \
  2>&1 | tee ~/snap/firefox/common/Q3DGStream_GPU_validation.log
```

The automated checks are:
1. **Real GPU inference**, loading the INT4 keep05 runtime checkpoint for NTC 114 and running tiny-cuda-nn on 1,024 active Gaussian coordinates from the saved compressed trajectory.
2. **Stage integrity**, checking all 1,556 unique records for NTC 114.
3. **Local QUIC receiver hooks**, checking that the *locally patched* viewer contains the QUIC transport and progressive receiver.
4. **299-frame parity**, checking all stage records against original compressed wire checkpoints.
5. **Three-frame CUDA evaluation**, running saliency plus five random orders over frames 113 to 115.

`RESULT: 5 PASS, 0 FAIL` establishes the offline GPU path and local receiver files are available. It does **not** prove real-time QUIC playback or image quality.

If this fails, send the full `Q3DGStream_GPU_validation.log` and do not reinstall CUDA or overwrite the viewer before diagnosis.

## Run the 30-update QUIC rendering pilot (manual visual check)

Only after the offline GPU tests pass, open **two different terminals**, both in `~/venvs/qntcstream`.

**Terminal A, receiver:**

```bash
cd ~/Q3DGStream-validation/research/quic/dynamic_priority_adapter
bash run_receiver.sh
```

**Terminal B, sender:**

```bash
cd ~/Q3DGStream-validation/research/quic/dynamic_priority_adapter
bash preflight.sh
QNTC_STAGE_DELAY=0 bash run_sender.sh
```

The receiver script **deletes only the dedicated cache** `/tmp/qntc_dynamic_priority_cache`. It requires previously generated `/tmp/qntc_progressive.crt` and `/tmp/qntc_progressive.key`, a saved fixed camera JSON, the bounded-v2 progressive sidecar, and the working color-preserving renderer patch. The scripts intentionally refuse to run if expected patches are absent.

Observe the chef moving in normal colors. Check that 30 NTC updates, 120 stage files and 30 addition files arrive. If the window freezes or colors change, preserve logs and inspect the receiver.

```bash
grep -E 'QUIC CPU READY|REAL QUIC ERROR|draw error|additions_' \
  /tmp/qntc_dynamic_priority_receiver.log | tail -60
tail -40 /tmp/qntc_dynamic_priority_sender.log
```

## Full 299-update test

Only when the 30-update pilot works, close the receiver/sender, relaunch receiver, then:

```bash
cd ~/Q3DGStream-validation/research/quic/dynamic_priority_adapter
QNTC_END=298 QNTC_STAGE_DELAY=0 bash run_sender.sh
```

Measure actual *applied* NTC updates/s, dropped frames, stage delays and renderer exceptions. A 30-FPS viewer display rate alone does not establish 30 NTC transformations per second.

## Reporting results

Please share:
- `~/snap/firefox/common/Q3DGStream_GPU_validation.log`
- `/tmp/qntc_dynamic_priority_receiver.log`
- `/tmp/qntc_dynamic_priority_sender.log`

Never upload QUIC TLS private keys or large scene/weight files into the Git repository. The official image-quality reference protocol must be tested separately before making PSNR/SSIM/LPIPS comparisons.
