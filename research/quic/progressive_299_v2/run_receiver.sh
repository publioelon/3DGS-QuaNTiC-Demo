#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
cd "$HOME/3DGS-QuaNTiC-Demo"
if [[ ! -s /tmp/qntc_progressive.crt || ! -s /tmp/qntc_progressive.key ]]; then
  echo '[STOP] QUIC local certificate missing. Follow README to create it.' >&2
  exit 1
fi
echo '[QUIC] Viewer: original colors, 300 display frames, no loop, CPU decode off UI thread'
LOOP_PLAYBACK=0 QNTC_PRIORITY_VIS=0 QNTC_PROGRESSIVE_QUIC=1 \
NTC_DEBUG=0 NTC_DEBUG_DXYZ=0 QNTC_PROGRESSIVE_CPU_CACHE=10 \
python -u main.py \
  --quic_listen 5002 \
  --quic_bind 127.0.0.1 \
  --quic_cache /tmp/qntc_progressive_cache \
  --quic_clear_cache \
  --quic_cert /tmp/qntc_progressive.crt \
  --quic_key /tmp/qntc_progressive.key \
  --frames 300 --autoplay --video_fps 30 \
  2>&1 | tee /tmp/qntc_receiver_v2.log
