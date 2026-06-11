# Linux setup and run guide

This guide explains how to run 3DGS-QuaNTiC on Linux with an NVIDIA GPU.

## 1. Install system packages

Run:

    sudo apt update
    sudo apt install -y git build-essential cmake ninja-build pkg-config libglfw3 libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-utils unzip wget

## 2. Clone the repository

Run:

    cd ~
    git clone https://github.com/publioelon/3DGS-QuaNTiC.git
    cd 3DGS-QuaNTiC

## 3. Create the Linux environment

For RTX 40-series GPUs:

    ./scripts/setup_linux_venv.sh
    source ~/venvs/qntcstream/bin/activate

For RTX 30-series GPUs:

    TCNN_CUDA_ARCHITECTURES=86 TORCH_CUDA_ARCH_LIST=8.6 ./scripts/setup_linux_venv.sh
    source ~/venvs/qntcstream/bin/activate

## 4. Download the demo scene

Run:

    cd ~/3DGS-QuaNTiC
    source ~/venvs/qntcstream/bin/activate
    ./scripts/download_flame_steak.sh

The scene is extracted to:

    ~/qntc_scenes/flame_steak_official

## 5. Run the receiver

Open terminal 1:

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

## 6. Run the sender

Open terminal 2:

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

## Hybrid NVIDIA laptops

If the viewer opens using integrated graphics instead of the NVIDIA GPU, run the receiver with NVIDIA PRIME:

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

Check the OpenGL renderer with:

    glxinfo -B | grep -E "OpenGL renderer|OpenGL version"

## Linux compatibility note

Linux support required a rasterizer compatibility change in renderer_cuda.py.

The Linux-tested setting is:

    "antialiasing": False,

The older setting below was removed because the Linux-installed diff-gaussian-rasterization version does not accept it:

    "bwd_depth": False,
