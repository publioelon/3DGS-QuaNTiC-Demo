#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
cd "$HOME/3DGS-QuaNTiC-Demo"
STAGE_DELAY="${QNTC_STAGE_DELAY:-0.045}"
echo "[QUIC] 299 compressed NTC updates, stage-delay=${STAGE_DELAY}s"
python -u qntc_progressive_send.py \
  --host 127.0.0.1 --port 5002 --insecure \
  --start 0 --end 298 --order saliency \
  --warmup-delay 2 --stage-delay "$STAGE_DELAY" \
  2>&1 | tee /tmp/qntc_sender_v2.log
