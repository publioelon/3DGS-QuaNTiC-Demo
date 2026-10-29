# FP32 original Flame Steak over native QUIC: correctness gate

**Purpose:** Check if full original FP32 scene (299 NTCs + all 299 additions) can be transferred over the project's existing QUIC file transport and displayed with the verified local rendering configuration. It does **not** use progressive updates, saliency, or compression. This deliberately separates source quality from transport correctness.

**Never overwrite the working renderer or original FP32 scene.** The package does not install viewer patches.

## Why the sender orders files this way

The native receiver may initialize its tiny-cuda-nn network when it gets `NTCs/config.json`. If the first NTC checkpoint hasn't arrived, the receiver can fall back to bounds computed from the initial Gaussian cloud, producing incorrect motion. So this sender sends `init_3dgs.ply`, followed by `NTC_000000.pth` and additions, **then** config, only after the configured prebuffer. Additions for frame N are sent before NTC N so the viewer never advances to N without the matching additions. With default prebuffer=299, the viewer starts only after all 299 updates are committed, maximizing fidelity validation, not real-time speed.

## Commands

Unzip into `~/snap/firefox/common`, then `cd ~/snap/firefox/common/QNTC_FP32_Reference_QUIC_V1`.

1. `source ~/venvs/qntcstream/bin/activate`
2. `python test_package.py`
3. `bash run_sender.sh --preflight-only` (read-only, no QUIC traffic; checks source and cache filesystem free space)
4. If certificate missing/expired: `openssl req -x509 -newkey rsa:2048 -nodes -days 7 -subj '/CN=localhost' -keyout /tmp/qntc_progressive.key -out /tmp/qntc_progressive.crt`
5. Terminal 1: `bash run_receiver.sh` (WARNING: clears only `/tmp/qntc_fp32_reference_cache`)
6. Terminal 2: `bash run_sender.sh` (full 299 FP32 NTCs and additions, buffered before playback)
7. Verify samples: `python verify_reference_cache.py --sample 12` (defaults `/tmp/qntc_fp32_reference_cache`; run from package dir)
8. When ready, verify all: `python verify_reference_cache.py`

The sender may transfer multiple GiB. No output is written under `/data` unless you explicitly select a cache there; `/data` was nearly full last time. It refuses to send when available cache disk space is insufficient. If needed, point to a filesystem with enough free storage: set `QNTC_FP32_CACHE=/path/to/dedicated_cache` in **both** terminals and use `--cache /path/to/dedicated_cache` for preflight/verify.

To reduce initial resource/time costs, use `bash run_sender.sh --frames 30 --prebuffer 30`; the viewer can only display the first 31 frames. Restart the receiver (clears cache) before the full test.

Once full transfer and fidelity are verified, streaming with prebuffer e.g. `--prebuffer 45` is a different stress test: receiver starts earlier but playback may pause if network throughput is insufficient. There is no artificial per-stage delay here.

## What to look for

- Native QUIC receiver selected because `QNTC_PROGRESSIVE_QUIC` is **unset**.
- Exact original NTC bounds printed: `[-20, -15, 5]` to `[15, 10, 23]` in earlier reference.
- 299 NTCs and 299 additions transmitted; native loader returns original trained FP32 weights.
- Same fixed camera, original SH coefficients, scale 1.0, rotation current, additions, no drift.
- If files verify but image differs, the issue is in live lazy loading / viewer state rather than QUIC file corruption or compression.

**Caveat:** Source `live_tcp.py` loader supports holding the previous NTC as a fallback, and GPU parsing of additions may affect rendering smoothness; this buffered correctness gate does not prove 30 FPS real-time streaming or dynamic-priority benefits. Do not claim it does.
