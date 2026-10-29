#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
VIEWER="${QNTC_VIEWER:-$HOME/3DGS-QuaNTiC-Demo}"
CACHE="${QNTC_FP32_CACHE:-/tmp/qntc_fp32_reference_cache}"
# Fail closed: this command uses --quic_clear_cache, which deletes the cache dir.
[[ "$(basename "$CACHE")" == 'qntc_fp32_reference_cache' ]] || {
  echo '[STOP] For safety, cache must be a dedicated folder named qntc_fp32_reference_cache' >&2
  exit 1
}
[[ "$CACHE" != "$VIEWER"/* && "$CACHE" != "$HOME/qntc_scenes/flame_steak_official"/* ]] || {
  echo '[STOP] Cache cannot be inside the source scene or viewer' >&2
  exit 1
}
cd "$VIEWER"
[[ -f camera_flame_steak.json ]] || { echo '[STOP] missing reference camera JSON' >&2; exit 1; }
[[ -f renderer_cuda.py ]] || { echo '[STOP] missing renderer' >&2; exit 1; }
# Preserve the working code only if no backups exist. Never overwrite earlier backups.
[[ -f renderer_cuda.py.WORKING_BACKUP ]] || cp -a renderer_cuda.py renderer_cuda.py.WORKING_BACKUP
[[ -f main.py.WORKING_BACKUP ]] || cp -a main.py main.py.WORKING_BACKUP
# Local-only test certificate. Require it to exist to avoid surprising prompts.
CERT=/tmp/qntc_progressive.crt
KEY=/tmp/qntc_progressive.key
[[ -s "$CERT" && -s "$KEY" ]] || { echo '[STOP] Missing local QUIC cert/key, see README' >&2; exit 1; }
echo '[RECEIVER] Native QUIC (NOT progressive sidecar), original FP32 NTCs and additions'
echo "[RECEIVER] Cache: $CACHE"
# The exact bounds come from NTC_000000, sent before config.json.
env -u STATE_SYNC_INTERVAL -u QNTC_PROGRESSIVE_QUIC -u QNTC_STATIC_CPU_PROBE \
  LOCK_CAMERA=1 FIXED_CAMERA_JSON="$VIEWER/camera_flame_steak.json" \
  NTC_TRANSLATION_SCALE=1.0 NTC_ROTATION_MODE=current \
  USE_NTC_MASK=0 LOOP_PLAYBACK=0 QNTC_PRIORITY_VIS=0 \
  NTC_DEBUG=0 NTC_DEBUG_DXYZ=0 \
  python -u main.py --quic_listen 5003 --quic_bind 127.0.0.1 \
  --quic_cache "$CACHE" --quic_clear_cache \
  --quic_cert "$CERT" --quic_key "$KEY" \
  --frames 300 --autoplay --video_fps 30 \
  2>&1 | tee /tmp/qntc_fp32_reference_receiver.log
