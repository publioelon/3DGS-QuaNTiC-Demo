# Architecture in one page

## What is represented?

- **3DGS:** scene as Gaussian positions, orientations, scales, opacity and appearance; rendering from new viewpoints.
- **Dynamic scene:** successive states of persistent Gaussians plus per-frame Gaussian additions.
- **NTC:** a per-step neural function: input Gaussian XYZ → multiresolution HashGrid → MLP → translation and quaternion update. Apply in the correct temporal sequence.
- **PQAT:** retain selected 64-weight blocks (e.g. 5%), train while simulating reduced precision, and encode retained INT4 values with one scale per block.
- **Saliency:** a predicted 2D attention heatmap from camera views. Map salient image regions back to spatial Gaussian groups.
- **Priority:** score retained NTC blocks for their influence on salient groups, send P0, P1, P2, P3 in that order.

```
original NTC .pth → sparse INT4 wire .pth
                  → rank each frame (SalFoM + gradient probes)
                  → stage P0.qblk | P1.qblk | P2.qblk | P3.qblk
                  → QUIC transport (application-level send order)
                  → decode received INT4 blocks
                  → construct partial NTC → predict Gaussian updates → render
```

## Critical correctness rule

At frame *t*, all progressive partial reconstructions must be evaluated against the **same Gaussian input state** for that timestep. Do not apply P0 displacement, then apply P1 displacement to the already-moved points. Instead, reconstruct each partial transform from the same pre-step state. If an approximate state is committed and carried into the next frame, errors can accumulate; sequential playback fidelity requires separate validation.

## What is an old NTC?

The frame index in the stage header identifies the model step. State lag is the difference between the target step and the most recently applied step, **not** the number of missing weights. One NTC per step is required for the current recurrent representation; receiving fewer than all its weights is a separate decision.

## What is actually sent?

Each `.qblk` stage contains a 14-byte header and multiple 40-byte weight records. QUIC transports these bytes; `.qblk` is our custom binary serialization, not a standard model format. In addition, the live receiver must have the initial PLY, NTC config/bounds and frame-specific additions.

## Research alternatives

A time-conditioned shared NTC or delta-NTC would address redundancy **between timesteps**. Saliency-aware delivery addresses **which information should arrive before a deadline**. These methods are complementary but the shared network is not implemented or benchmarked in the current artifacts.
