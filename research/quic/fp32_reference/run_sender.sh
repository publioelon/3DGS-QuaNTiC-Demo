#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
VIEWER="${QNTC_VIEWER:-$HOME/3DGS-QuaNTiC-Demo}"
CACHE="${QNTC_FP32_CACHE:-/tmp/qntc_fp32_reference_cache}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$VIEWER${PYTHONPATH:+:$PYTHONPATH}"
python -u "$HERE/fp32_quic_sender.py" --cache "$CACHE" "$@" \
  2>&1 | tee /tmp/qntc_fp32_reference_sender.log
