# QNTC-Stream

This repository contains a small experimental viewer and TCP streaming setup for dynamic 3D Gaussian Splatting scenes based on the 3DGStream representation.

The demo sends the initial 3D Gaussian scene once and then progressively streams Neural Transformation Cache (NTC) files. The NTCs update the motion of the scene over time. This version is focused on the compression and progressive-delivery part of the idea, so TCP is used as a stable baseline transport.

## Main files

`main.py` starts the viewer. It supports offline loading and live TCP receiving.

`live_tcp.py` receives streamed scene files, writes them into a local cache, and exposes the received NTCs to the renderer.

`tcp_fvv_sender.py` sends the initial scene, the NTC config, and the selected NTC files to the viewer.

`renderer_cuda.py` applies the NTC motion and renders the dynamic Gaussian scene.

`util_3dgstream.py` contains the loading code for FP32, FP16, INT8, and INT4 NTC checkpoints.

## Expected scene layout

A streamed scene should follow this structure:

```text
scene_root/
  init_3dgs.ply
  NTCs/
    config.json
    NTC_000000.pth
    NTC_000001.pth
    ...
```

The current stable demo does not require `additional_3dgs`.

## Local TCP test

Open two Anaconda Prompt terminals.

Receiver:

```bat
conda activate 3dgstream
cd /d C:\Users\Publi\QNTC-Stream

python main.py ^
  --tcp_listen 5001 ^
  --tcp_bind 127.0.0.1 ^
  --tcp_cache C:\tmp\qntc_stream_test ^
  --tcp_clear_cache ^
  --frames 300 ^
  --autoplay
```

Sender:

```bat
conda activate 3dgstream
cd /d C:\Users\Publi\QNTC-Stream

python tcp_fvv_sender.py ^
  --host 127.0.0.1 ^
  --port 5001 ^
  --root "C:\path\to\scene_root" ^
  --start 0 ^
  --end 298 ^
  --ntc_stride 1 ^
  --no_additions
```

Sparse update-rate test:

```bat
python tcp_fvv_sender.py ^
  --host 127.0.0.1 ^
  --port 5001 ^
  --root "C:\path\to\scene_root" ^
  --start 0 ^
  --end 298 ^
  --ntc_stride 5 ^
  --no_additions
```

## Notes

The scene files are intentionally not included here. Keep large `.ply` and `.pth` files outside the repository, or provide them through a separate dataset/download link.

The `shaders/` folder contains a compact OpenGL fallback shader pair. If you want exact visual parity with your previous local project, you can replace them with the original `shaders/gau_vert.glsl` and `shaders/gau_frag.glsl` from your working `C:\Users\Publi\QNTC-Stream` folder.
