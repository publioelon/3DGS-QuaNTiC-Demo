# 3DGS-QuaNTiC

**Live Streaming Dynamic 3D Gaussian Splatting Scenes with Quantized Neural Transformation Caches**

3DGS-QuaNTiC is a live streaming demo framework for dynamic 3D Gaussian Splatting scenes represented with Neural Transformation Caches (NTCs).

The system sends the initial 3D Gaussian scene once and then progressively streams NTC files to update the motion of the scene over time. This repository focuses on compressed NTC delivery and progressive playback using TCP as a stable baseline transport.

## Features

- Live TCP streaming of dynamic 3D Gaussian Splatting scenes.
- Initial 3DGS scene transmission followed by progressive NTC updates.
- Support for sparse NTC update streaming through the `ntc_stride` option.
- Receiver-side file cache for progressive playback.
- Linux/Ubuntu support with PyTorch CUDA, tiny-cuda-nn, and diff-gaussian-rasterization.
- Runtime stream monitor showing received files, goodput, cache status, and playback state.

## Main files

- `main.py`: viewer and TCP receiver entry point.
- `live_tcp.py`: receives streamed scene files, writes them to the local cache, and exposes received NTCs to the renderer.
- `tcp_fvv_sender.py`: sends the initial scene, NTC config, and selected NTC files to the viewer.
- `renderer_cuda.py`: applies NTC motion and renders the dynamic Gaussian scene.
- `renderer_ogl.py`: OpenGL rendering path used by the viewer.
- `util_3dgstream.py`: 3DGStream scene loading utilities.
- `NTC.py`: Neural Transformation Cache model definition.
- `scripts/setup_linux_venv.sh`: creates the Linux Python environment and installs CUDA extensions.
- `scripts/download_flame_steak.sh`: downloads the Flame Steak demo scene.
- `requirements-linux.txt`: Linux Python runtime dependencies that do not require custom CUDA compilation.

## Tested platform

The Linux version was tested on Ubuntu with an NVIDIA GPU.

The setup script defaults to RTX 40-series settings:

    TCNN_CUDA_ARCHITECTURES=89
    TORCH_CUDA_ARCH_LIST=8.9

For RTX 30-series GPUs, use:

    TCNN_CUDA_ARCHITECTURES=86
    TORCH_CUDA_ARCH_LIST=8.6

## 1. Install Ubuntu system packages

Run:

    sudo apt update
    sudo apt install -y git build-essential cmake ninja-build pkg-config libglfw3 libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-utils unzip wget

## 2. Clone the repository

After the repository is renamed, use:

    cd ~
    git clone https://github.com/publioelon/3DGS-QuaNTiC.git
    cd 3DGS-QuaNTiC

If you are using the old repository name before the rename, use:

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

The setup script installs:

- PyTorch CUDA 12.6
- cuda-python
- tiny-cuda-nn
- diff-gaussian-rasterization
- simple-knn
- GLFW / OpenGL Python viewer dependencies

## 4. Download the Flame Steak demo scene

Run:

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

The scene files are not stored in this repository because `.ply` and `.pth` files are large.

## 5. Run the TCP demo

Open two terminals.

### Terminal 1: receiver/viewer

Run:

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

If you are still using the old local folder name, replace:

    cd ~/3DGS-QuaNTiC

with:

    cd ~/QNTC-Stream-Demo

### Terminal 2: sender

Run:

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

To stream fewer NTC updates, increase `ntc_stride`.

Example:

    python tcp_fvv_sender.py \
      --host 127.0.0.1 \
      --port 5001 \
      --root "$HOME/qntc_scenes/flame_steak_official" \
      --start 0 \
      --end 298 \
      --ntc_stride 5 \
      --no_additions

With stride-based streaming, only every N-th NTC is transmitted. The receiver reuses the most recently available NTC when an intermediate update is unavailable.

## Hybrid NVIDIA laptops

On some laptops, OpenGL may open on the integrated GPU instead of the NVIDIA GPU. Check with:

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

## Troubleshooting

### GitHub scene download fails

If `gdown` fails, open `scripts/download_flame_steak.sh` and verify that the Google Drive file ID is still valid.

### Viewer opens but stays black

First check the viewer panel. If it reports a rasterizer argument error, the installed `diff-gaussian-rasterization` version may not match the expected API.

Also check that the receiver has received:

    init_3dgs.ply
    NTCs/config.json
    NTCs/NTC_000000.pth

### OpenGL uses the wrong GPU

Use the NVIDIA PRIME receiver command shown above.

### simple-knn reports `libc10.so` not found

Import `torch` before importing `simple_knn`, or ensure PyTorch shared libraries are visible in the active environment.

## Acknowledgment

This project builds on the 3DGStream/3DGStreamViewer codebase and extends it with TCP-based progressive streaming, quantized NTC loading, sparse NTC update support, and live playback/cache monitoring.
