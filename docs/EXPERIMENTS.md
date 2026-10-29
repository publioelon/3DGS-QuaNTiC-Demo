# Reproducing the experiments

The `research/` tree contains the original experimental scripts and setup notes. Large NTC checkpoints, Gaussian scenes, Gaussian states, SalFoM prediction maps, priority stage outputs and full evaluation traces are **external inputs** rather than files checked into Git.

## 1. Priority generation

```bash
cd research/priority/dynamic_299
python3 test_offline.py
# Requires generated scene data and tiny-cuda-nn for ranking:
bash run_pilot.sh
bash run_small_sequence.sh
bash run_all.sh
```

The generator combines measured/interpolated 12-camera SalFoM outputs with per-NTC gradient-based block sensitivity. Four `.qblk` stages are saved for each compressed checkpoint.

## 2. Stage integrity

```bash
python3 q3dgstream.py inspect-stage /path/to/NTC_000114_P0.qblk
python3 research/evaluation/qntc_299_priority_eval.py --verify-only
```

The second command expects the original PQAT checkpoint and stage directories. It confirms the correspondence of indices, scales and packed INT4 weights for each frame.

## 3. Reconstruction accuracy

```bash
python3 research/evaluation/qntc_299_priority_eval.py \
  --start 113 --end 115 --seeds 5 --batch 8192

python3 research/evaluation/qntc_299_priority_eval.py \
  --start 0 --end 298 --seeds 5 --batch 8192
```

These measurements compare per-frame partial NTC inference with the **complete compressed NTC** for each saved input Gaussian state. They do not accumulate partial transforms across frames. The `rotation_v2` evaluator additionally checks the zero-quaternion behavior using an explicit fallback convention; ensure your viewer matches that convention before drawing conclusions about playback rotation.

## 4. Inspect the sample results

`research/results/` contains a small three-frame example and the 299-stage integrity report. The full 299-frame quality CSV is stored separately to avoid coupling a code checkout to research outputs.

## Results protocols

The previously validated FP32 reference of **33.938 dB, SSIM 0.9591, LPIPS 0.0776** was measured with an older **50-frame** protocol. The separate later lowbit299 experiment evaluated 299 updates under a different renderer protocol. Do not compare absolute image-quality metrics between these protocols as if they shared an identical baseline. Details are in [Results](RESULTS.md).
