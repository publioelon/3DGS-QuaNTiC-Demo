# Installation and quickstart

Q3DGStream supports two distinct workflows: a **dependency-free binary format demo** and the **GPU-enabled viewer**.

## Step 1: Clone and check the source

```bash
git clone https://github.com/publioelon/3DGS-QuaNTiC-Demo.git
cd 3DGS-QuaNTiC-Demo
python3 q3dgstream.py demo
python3 q3dgstream.py test
python3 q3dgstream.py doctor
```

`demo` creates and parses an in-memory synthetic `.qblk` stage without requiring model weights or CUDA. The tests validate QNP01 packing and source syntax. `doctor` reports missing optional GPU dependencies and scene data.

## Step 2: Install NVIDIA / CUDA tools

For Ubuntu with an NVIDIA GPU, follow [Linux setup](linux.md), or run the repository installer:

```bash
./scripts/setup_linux_venv.sh
source ~/venvs/qntcstream/bin/activate
./scripts/download_flame_steak.sh
python3 q3dgstream.py doctor
```

The installer builds CUDA extensions, including tiny-cuda-nn and the Gaussian rasterizer. GPU compute capability and installed CUDA/PyTorch versions must match the machine. The scene downloader places `init_3dgs.ply` and 299 NTC checkpoints under `~/qntc_scenes/flame_steak_official/`.

## Step 3: Run the existing TCP baseline

Open two terminals with the virtual environment active.

**Terminal A (viewer):**

```bash
python main.py --tcp_listen 5001 --tcp_bind 127.0.0.1 \
  --tcp_cache /tmp/q3dgstream_tcp --tcp_clear_cache --frames 300 \
  --video_fps 30 --autoplay
```

**Terminal B (sender):**

```bash
python tcp_fvv_sender.py --host 127.0.0.1 --port 5001 \
  --root "$HOME/qntc_scenes/flame_steak_official" \
  --start 0 --end 298 --ntc_stride 1 --no_additions
```

This is the existing viewer's TCP baseline, **not** the experimental QUIC priority path. See the [QUIC guide](QUIC.md) for the prototype.

## Step 4: Check a real quantized stage

If the stages have already been generated:

```bash
python3 q3dgstream.py inspect-stage \
  ~/snap/firefox/common/QNTC_dynamic_299_priorities_v1/stages/NTC_000114_P0.qblk
```

## Step 5: Run the offline research evaluation

See [Experiments](EXPERIMENTS.md). Stage generation and model inference require the original large datasets, the PQAT checkpoint and tested GPU dependencies. Those data files are not included in Git.
