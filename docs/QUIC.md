# QUIC streaming and partial NTC reconstruction

The QUIC implementation under `research/quic/` is a research prototype, separate from the stable TCP player at the repository root.

## Available experiments

| Directory | Purpose |
|---|---|
| `research/quic/fp32_reference` | Baseline FP32 QUIC sender and cache verification |
| `research/quic/progressive_pilot_v1` | Early P0–P3 receiver and sender integration |
| `research/quic/progressive_299_v2` | Bounded progressive state-cache implementation |
| `research/quic/dynamic_priority_adapter` | Sends per-frame saliency-ranked `.qblk` stages with Gaussian additions |

## Required receiver implementation

The sender scripts are included in Git, but the GPU-tested viewer contains local modifications to `main.py`, `renderer_cuda.py` and the progressive receiver sidecar. **A clone of the original `main` tree cannot yet run the updated QUIC pipeline as-is.** Integrate and test those local viewer changes before relying on these scripts for real-time QUIC demos.

Once the receiver is integrated, the typical pilot launches the receiver first, then the sender:

```bash
cd research/quic/dynamic_priority_adapter
bash preflight.sh
# Terminal A:
bash run_receiver.sh
# Terminal B:
bash run_sender.sh
```

The pilot sends **30** frame updates by default. Its `.qblk` streams are ordered by application-level priority, not by QUIC-native scheduling. The full 299-update path and adaptive streaming under network dynamics are distinct validation steps.

## Important state semantics

A `.qblk` chunk can be decoded independently, but it is a block of **neural network parameters**, not a tile of rendered geometry. To render after a partial update, assemble the currently available parameters and evaluate the NTC against the *same pre-step Gaussian state*. Reevaluation after P1 must not double-apply the motion already produced by P0.

For a 30 Hz animation, the server sends one per-frame NTC update every ~33.3 ms in the existing model. **NTC age** is the difference between intended playback step and last committed step. **Partial reception** is how many P0–P3 records of the current step have arrived. These two quantities are different.

## Next integration

The newer receiver and CUDA-renderer patches must be checked against the latest local working copy, committed with reproducible smoke tests, and then measured under realistic QUIC latency, loss and throughput constraints. See [Results](RESULTS.md) for the currently validated subset.
