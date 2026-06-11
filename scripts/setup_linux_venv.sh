#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${HOME}/venvs/qntcstream}"

echo "[QNTC] Repository: $REPO_DIR"
echo "[QNTC] Virtual environment: $VENV"

python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "[QNTC] Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip "setuptools==80.9.0" wheel packaging

echo "[QNTC] Installing PyTorch CUDA 12.6..."
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "[QNTC] Installing viewer dependencies..."
python -m pip install -r "$REPO_DIR/requirements-linux.txt"

echo "[QNTC] Installing tiny-cuda-nn..."
export CUDA_HOME="${CUDA_HOME:-/usr}"
export MAX_JOBS="${MAX_JOBS:-4}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"

python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

echo "[QNTC] Installing Gaussian Splatting CUDA extensions..."
if [ ! -d "${HOME}/gaussian-splatting" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "${HOME}/gaussian-splatting"
fi

cd "${HOME}/gaussian-splatting"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/simple-knn

echo ""
echo "[QNTC] Linux environment ready."
echo "Activate it with:"
echo "source ${VENV}/bin/activate"
