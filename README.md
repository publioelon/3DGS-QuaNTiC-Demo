# 3DGS-QuaNTiC

**Live Streaming Dynamic 3D Gaussian Splatting Scenes with Quantized Neural Transformation Caches**

3DGS-QuaNTiC is a live streaming demo framework for dynamic 3D Gaussian Splatting scenes represented with Neural Transformation Caches (NTCs).

The system sends the initial 3D Gaussian scene once and then progressively streams NTC files to update the motion of the scene over time. This repository focuses on compressed NTC delivery and progressive playback using TCP as a stable baseline transport.

## Features

- Live TCP streaming of dynamic 3D Gaussian Splatting scenes.
- Initial 3DGS scene transmission followed by progressive NTC updates.
- Sparse NTC update streaming through the `--ntc_stride` option.
- Receiver-side cache for progressive playback.
- Linux/Ubuntu support with PyTorch CUDA, `tiny-cuda-nn`, and `diff-gaussian-rasterization`.
- Runtime stream monitor for received files, goodput, cache status, and playback state.

## Main files

- `main.py`: viewer and TCP receiver entry point.
- `live_tcp.py`: receives streamed scene files and writes them to the local cache.
- `tcp_fvv_sender.py`: sends the initial scene, NTC config, and selected NTC files.
- `renderer_cuda.py`: applies NTC motion and renders the dynamic Gaussian scene.
- `renderer_ogl.py`: OpenGL rendering path used by the viewer.
- `util_3dgstream.py`: 3DGStream scene loading utilities.
- `NTC.py`: Neural Transformation Cache model definition.
- `requirements-linux.txt`: Linux Python dependencies.
- `scripts/setup_linux_venv.sh`: creates the Linux Python environment and installs CUDA extensions.
- `scripts/download_flame_steak.sh`: downloads the Flame Steak demo scene.

## Tested platform

The Linux version was tested on Ubuntu with an NVIDIA GPU.

The default setup targets RTX 40-series GPUs:

    TCNN_CUDA_ARCHITECTURES=89
    TORCH_CUDA_ARCH_LIST=8.9

For RTX 30-series GPUs, use:

    TCNN_CUDA_ARCHITECTURES=86
    TORCH_CUDA_ARCH_LIST=8.6

## 1. Install Ubuntu system packages

    sudo apt update
    sudo apt install -y git build-essential cmake ninja-build pkg-config libglfw3 libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-utils unzip wget

## 2. Clone the repository

After the repository is renamed:

    cd ~
    git clone https://github.com/publioelon/3DGS-QuaNTiC.git
    cd 3DGS-QuaNTiC

Before the repository rename:

    cd ~
    git clone https://github.com/publioelon/QNTC-Stream-Demo.git
    cd QNTC-Stream-Demo

## 3. Create the Linux Python environment

For RTX 40-series GPUs:

    ./scripts/setup_linux_venv.sh
    source ~/venvs/qntcstream/bin/activate

For RTX 30-series GPUs:

    TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 ./scripts/setup_linux_venv.sh
    source ~/venvs/qntcstream/bin/activate

The setup script installs PyTorch CUDA, `cuda-python`, `tiny-cuda-nn`, `diff-gaussian-rasterization`, `simple-knn`, and the OpenGL viewer dependencies.

## 4. Download the Flame Steak demo scene

    source ~/venvs/qntcstream/bin/activate
    ./scripts/download_flame_steak.sh

The scene is extracted to:

    ~/qntc_scenes/flame_steak_official

Expected structure:

    flame_steak_official/
      init_3dgs.ply
      NTCs/
        config.json
        NTC_000000.pth
        ...
        NTC_000298.pth

Scene files are not stored in this repository because `.ply` and `.pth` files are large.

## Quantizing NTC files

The repository includes `quantize_ntcs.py`, which converts NTC files to FP16, INT8, or INT4.

FP16 example:

    python quantize_ntcs.py \
      --src "$HOME/qntc_scenes/flame_steak_official" \
      --out "$HOME/qntc_scenes/flame_steak_fp16" \
      --mode fp16 \
      --overwrite \
      --verify

INT8 example:

    python quantize_ntcs.py \
      --src "$HOME/qntc_scenes/flame_steak_official" \
      --out "$HOME/qntc_scenes/flame_steak_int8_b64" \
      --mode int8 \
      --block-size 64 \
      --overwrite \
      --verify

INT4 example:

    python quantize_ntcs.py \
      --src "$HOME/qntc_scenes/flame_steak_official" \
      --out "$HOME/qntc_scenes/flame_steak_int4_b64" \
      --mode int4 \
      --block-size 64 \
      --overwrite \
      --verify

The script writes `quantization_manifest.json` with file sizes, compression ratio, reduction percentage, and optional reconstruction-error statistics.

In the Flame Steak test scene, the observed NTC reductions were approximately:

- FP16: 49.98%
- INT8, block size 64: 74.18%
- INT4, block size 64: 86.68%

FP16 outputs are saved as regular PyTorch tensors and can be streamed directly with the current viewer. INT8 and INT4 outputs are saved as packed quantized blobs and require loader-side dequantization support before direct rendering.


## 5. Run the TCP demo

Open two terminals.

### Terminal 1: receiver/viewer

    cd ~/3DGS-QuaNTiC
    source ~/venvs/qntcstream/bin/activate

    python main.py \
      --tcp_listen 5001 \
      --tcp_bind 127.0.0.1 \
      --tcp_cache /tmp/qntc_stream_test \
      --tcp_clear_cache \
      --frames 300 \
      --video_fps 30 \
      --autoplay

If your local folder still has the old name, use:

    cd ~/QNTC-Stream-Demo

### Terminal 2: sender

    cd ~/3DGS-QuaNTiC
    source ~/venvs/qntcstream/bin/activate

    python tcp_fvv_sender.py \
      --host 127.0.0.1 \
      --port 5001 \
      --root "$HOME/qntc_scenes/flame_steak_official" \
      --start 0 \
      --end 298 \
      --ntc_stride 1 \
      --no_additions

The viewer should display the dynamic Flame Steak scene while the sender streams the initial 3DGS and NTC files.

## Sparse NTC update mode

To stream fewer NTC updates, increase `--ntc_stride`.

Example:

    python tcp_fvv_sender.py \
      --host 127.0.0.1 \
      --port 5001 \
      --root "$HOME/qntc_scenes/flame_steak_official" \
      --start 0 \
      --end 298 \
      --ntc_stride 5 \
      --no_additions

With stride-based streaming, only every N-th NTC is transmitted. The receiver reuses the most recently received NTC for intermediate frames.

## Hybrid NVIDIA laptops

On some laptops, OpenGL may open on the integrated GPU instead of the NVIDIA GPU.

Check with:

    glxinfo -B | grep -E "OpenGL renderer|OpenGL version"

If the renderer is not NVIDIA, run the receiver with NVIDIA PRIME:

    __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    python main.py \
      --tcp_listen 5001 \
      --tcp_bind 127.0.0.1 \
      --tcp_cache /tmp/qntc_stream_test \
      --tcp_clear_cache \
      --frames 300 \
      --video_fps 30 \
      --autoplay

## Linux compatibility note

Linux support required a rasterizer compatibility change in `renderer_cuda.py`.

The Linux-tested setting is:

    "antialiasing": False,

The older setting below was removed because the Linux-installed `diff-gaussian-rasterization` version does not accept it:

    "bwd_depth": False,

## Acknowledgment

This project builds on the 3DGStream/3DGStreamViewer codebase and extends it with TCP-based progressive streaming, quantized NTC loading, sparse NTC update support, and live playback/cache monitoring.
