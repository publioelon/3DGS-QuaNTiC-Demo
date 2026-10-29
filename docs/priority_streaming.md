# What the system does

## Models and motion

**Dynamic 3DGS** stores a set of persistent 3D Gaussians that evolve with time, plus additional Gaussians where the source requires them. An **NTC** (Neural Transformation Cache) predicts the per-step Gaussian translation and quaternion rotation update from current 3D positions. Our current architecture evaluates one NTC per animation update on the **receiver**; the server does not transmit all fully transformed Gaussians at each frame.

The NTC uses a **multiresolution HashGrid** to turn input XYZ positions into learned spatial features, then an MLP predicts the updates.

## Compression

**PQAT** combines block pruning and low-bit quantization aware training. The selected `int4_keep05` model retains 1,556 of 31,120 64-value blocks (5%) and stores each retained block as 32 bytes of packed 4-bit values plus a per-block scale and block index. A representative checkpoint is 66,741 bytes in the PyTorch **wire checkpoint serialization**. The custom `.qblk` stage serialization has a different byte count because it omits some checkpoint metadata.

## Saliency-aware priority

A 12-camera SalFoM pilot produced image-space attention maps. The dynamic priority pipeline combines multi-view saliency information with per-frame NTC block sensitivity, assigns priorities and saves **four binary stages** per NTC, P0 to P3. Stage data is not directly renderable geometry; it modifies sparse neural network parameters.

Receive P0 → reconstruct a **partial NTC** → infer update from the frame's original input Gaussian state → draw. When P1 arrives, reconstruct from **the same pre-step state** using P0+P1. Repeat through P3. Never compound intermediate stage transforms on an already partially moved state.

## How do we know if the state is old?

Each stage includes the NTC frame index. State lag in frames = **intended playback step − last applied complete/committed step**. At 30 animation steps/s, three steps correspond to 100 ms of animation. The age metric is distinct from the fraction of weights received for a particular NTC.

## QUIC scope

The pilot sends separate reliable files as QUIC application-level streams in priority order. It is **not** QUIC transport-native priority scheduling, nor a QoE study under network RTT variation. Adding a real priority scheduler and measuring real rendering deadlines remains an integration task.

## What about shared NTCs?

A time-conditioned network shared across frames, or delta transmission between consecutive NTCs, might reduce temporal redundancy. These are **additional, unimplemented** research directions. Shared NTCs do not replace saliency-aware prioritization, because they optimize a different part of the problem.

See `UNDERGRADUATE_START_HERE.md` for a practical onboarding sequence.
