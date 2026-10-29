# Verified evidence and measurement boundaries

## What passed

| Test | Observation |
|---|---|
| Compressed student trajectory | All 300 Gaussian states replayed, and all 14 available reference checkpoints matched exactly for XYZ and stored quaternion components |
| `.qblk` integrity | 299/299 stage sets matched the original INT4 compressed checkpoint records (indices, per-block FP32 scales, packed 4-bit values) |
| Progressive evaluation | 299 frames, one saliency schedule and five independent random schedules, stages P0 to P3 |
| Basic QUIC pilot | 30 frames; 120 stages and 30 addition files received; application-level priority send order |
| HashGrid mapping | NTC 114 structural lookup checked against 12 sampled tiny-cuda-nn gradients |

## Relative translation RMS (saliency compared with five random orders)

At roughly 75% cumulative block arrival, saliency ordering reduced error by ~76.75% in the red group and ~78.28% in the orange group relative to the random baseline mean. At 100% block reception the translation output matched the full compressed source. The evaluated source was **INT4 keep05**, not the uncompressed FP32 teacher.

These are *isolated per-frame transformation* metrics, not yet end-to-end perceptual metrics under network loss/congestion. Current v1 rotation summaries can omit zero-length quaternion vectors; the v2 script uses an **explicit identity fallback** convention that needs checking against the live viewer's normalization behavior.

## Image-quality metrics are not all on the same protocol

Earlier Flame Steak PQAT results referenced FP32 PSNR 33.938, SSIM 0.9591, LPIPS 0.0776, assessed over **50 frames** in the saved all-scenes spreadsheet. The later lowbit299 run covers **299 updates**, with an approximately 30.612 dB FP32 renderer/protocol reference. Do not combine their PSNR/SSIM/LPIPS differences as a single apples-to-apples ranking until rendered under one agreed protocol.

## Transmission overhead

For a representative keep05 NTC, the PyTorch packed wire checkpoint was 66,741 bytes. Four `.qblk` stage payloads contain 1,556 × 40-byte records plus 4 × 14-byte stage headers = 62,296 bytes. These are different serializations of identical INT4 block content. The `.qblk` files exclude scene initialization, additions, connection overhead, and some checkpoint metadata. Report bitrates using the explicitly chosen payload basis.

## Priorities still need network-level validation

The QUIC pilot demonstrates real local QUIC file delivery, **not** transport-internal stream prioritization and not an adaptive throughput/RTT-controlled QoE test. Stage delays used to visualize progression lower the achieved update rate by design. Full 299-frame live playback and end-to-end saliency-weighted image-quality comparisons are separate milestones.

## Uncollected local source

The exact GPU-tested QUIC viewer files and `replay_int4_keep05_299.py` were executed on the Ubuntu GPU laptop but are not contained in these recovered conversation attachments. They must be collected from that machine, reviewed and checked in separately.
