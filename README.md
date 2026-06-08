# QNTC-Stream

QNTC-Stream is a small TCP-based streaming demo for dynamic 3D Gaussian Splatting scenes represented with Neural Transformation Caches (NTCs).

The system sends the initial 3D Gaussian scene once and then progressively streams NTC files to update the motion of the scene over time. This repository focuses on compressed NTC delivery and progressive playback, so TCP is used as a stable baseline transport.

## Main files

`main.py` starts the viewer. It supports offline loading and live TCP receiving.

`live_tcp.py` receives streamed scene files, writes them into a local cache, and exposes the received NTCs to the renderer.

`tcp_fvv_sender.py` sends the initial scene, the NTC config, and the selected NTC files to the viewer.

`renderer_cuda.py` applies the NTC motion and renders the dynamic Gaussian scene.

`renderer_ogl.py` provides the OpenGL rendering path used by the viewer.

`util_3dgstream.py` contains the loading code for FP32, FP16, INT8, and INT4 NTC checkpoints.

`util_gau.py` loads the initial Gaussian PLY scene.

`NTC.py` defines the Neural Transformation Cache wrapper used during playback.

## Scene files

The scene files are not included in this repository because they contain large `.ply` and `.pth` files.

A streamed scene should follow this structure:

```text
scene_root/
  init_3dgs.ply
  NTCs/
    config.json
    NTC_000000.pth
    NTC_000001.pth
    NTC_000002.pth
    ...
```

The current stable demo does not require `additional_3dgs`.

A ready-to-run demo scene can be downloaded from:

```text
TODO: add demo scene download link here
```

After downloading and extracting the scene package, use the extracted folder as the `--root` argument when running `tcp_fvv_sender.py`.

To obtain scenes from scratch, follow the original 3DGStream instructions: https://github.com/SJoJoK/3DGStream

## Environment

This project requires a CUDA-capable Python environment. The code was tested using a Conda environment named `3dgstream`.

Typical dependencies include:

```text
PyTorch with CUDA
tiny-cuda-nn
diff-gaussian-rasterization
cuda-python
glfw
PyOpenGL
imgui
plyfile
imageio
PyGLM
numpy
```

Activate your environment before running the viewer or sender:

```bat
conda activate 3dgstream
```

If the basic Python packages are missing, they can be installed with:

```bat
pip install glfw PyOpenGL imgui plyfile imageio PyGLM
```

The CUDA-specific packages, especially `tiny-cuda-nn` and `diff-gaussian-rasterization`, should match the CUDA and PyTorch versions installed on your machine.

## Local TCP test

Open two Anaconda Prompt terminals.

Replace:

```text
C:\path\to\QNTC-Stream-Demo
```

with the folder where this repository was cloned, and replace:

```text
C:\path\to\scene_root
```

with the folder containing `init_3dgs.ply` and the `NTCs/` directory.

### Terminal 1: receiver/viewer

```bat
conda activate 3dgstream
cd /d C:\path\to\QNTC-Stream-Demo

python main.py ^
  --tcp_listen 5001 ^
  --tcp_bind 127.0.0.1 ^
  --tcp_cache C:\tmp\qntc_stream_test ^
  --tcp_clear_cache ^
  --frames 300 ^
  --autoplay
```

This starts the viewer and waits for the sender.

### Terminal 2: sender

```bat
conda activate 3dgstream
cd /d C:\path\to\QNTC-Stream-Demo

python tcp_fvv_sender.py ^
  --host 127.0.0.1 ^
  --port 5001 ^
  --root "C:\path\to\scene_root" ^
  --start 0 ^
  --end 298 ^
  --ntc_stride 1 ^
  --no_additions
```

Use `--ntc_stride 1` to send every NTC file.

For a sparse update-rate test, use a larger stride:

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

With `--ntc_stride 5`, the sender transmits one NTC every five frames. The receiver keeps playback running by reusing the most recent received NTC for the intermediate frames.

## Expected output

On the sender side, a successful run should show messages similar to:

```text
[SENDER] selected NTC count: 299
[SENDER] connected to 127.0.0.1:5001
[SEND] init_3dgs.ply
[SEND] NTCs/config.json
[SEND] NTCs/NTC_000000.pth
...
[SENDER] done
```

For a 299-NTC scene, `--ntc_stride 5` should select around 60 NTC files:

```text
NTC_000000.pth
NTC_000005.pth
NTC_000010.pth
...
NTC_000295.pth
```

On the receiver side, the viewer should open and print NTC/rendering diagnostics while playback runs.

## Notes

TCP is used here as a simple and stable transport for testing progressive delivery of compressed NTC updates. More advanced transport options, such as RTP/UDP or QUIC, are left for future work.

Large scene files such as `.ply`, `.pth`, `.pt`, and `.ckpt` should not be committed directly to the Git repository. Store them externally and provide a download link instead.

## Acknowledgment

This project builds on the 3DGStream/3DGStreamViewer codebase and extends it with TCP-based progressive streaming, quantized NTC loading, sparse NTC update support, and live playback/cache monitoring.
