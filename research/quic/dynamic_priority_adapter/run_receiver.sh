#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
VIEWER="$HOME/3DGS-QuaNTiC-Demo"
CACHE=/tmp/qntc_dynamic_priority_cache
[[ ! -L "$CACHE" ]] || { echo '[STOP] Cache path is a symlink' >&2; exit 1; }
[[ -s /tmp/qntc_progressive.crt && -s /tmp/qntc_progressive.key ]] || {
  echo '[STOP] Missing local /tmp/qntc_progressive.{crt,key} files' >&2; exit 1;
}
[[ -f "$VIEWER/camera_flame_steak.json" ]] || { echo '[STOP] Missing fixed camera' >&2; exit 1; }
[[ -f "$VIEWER/qntc_progressive_live.py" ]] || { echo '[STOP] Missing progressive receiver sidecar' >&2; exit 1; }
[[ -f "$VIEWER/renderer_cuda.py.WORKING_BACKUP" ]] || { echo '[STOP] Save working renderer as renderer_cuda.py.WORKING_BACKUP first' >&2; exit 1; }
cd "$VIEWER"
python - <<'PY'
from pathlib import Path
s=Path('renderer_cuda.py').read_text()
if 'if self.qntc_priority_vis:' not in s or 'sh_vis = rendered_gaussians.sh' not in s:
    raise SystemExit('[STOP] Working original-SH color bypass not detected; do not run')
q=Path('qntc_progressive_live.py').read_text()
if 'part_versions' not in q or 'decode_stages' not in q:
    raise SystemExit('[STOP] Expected bounded progressive v2 receiver not installed')
print('[OK] Working renderer color fix and bounded progressive receiver detected')
PY
# Clears only /tmp/qntc_dynamic_priority_cache. Never the FP32 validated cache.
echo '[RECEIVER] Dynamic priority QUIC pilot on UDP 5002; local-only'
env -u STATE_SYNC_INTERVAL -u QNTC_STATIC_CPU_PROBE \
    LOCK_CAMERA=1 FIXED_CAMERA_JSON="$VIEWER/camera_flame_steak.json" \
    NTC_TRANSLATION_SCALE=1.0 NTC_ROTATION_MODE=current \
    USE_NTC_MASK=0 QNTC_PRIORITY_VIS=0 QNTC_PROGRESSIVE_QUIC=1 \
    LOOP_PLAYBACK=1 QNTC_PROGRESSIVE_CPU_CACHE=10 \
    NTC_DEBUG=0 NTC_DEBUG_DXYZ=0 \
    python -u main.py --quic_listen 5002 --quic_bind 127.0.0.1 \
      --quic_cache "$CACHE" --quic_clear_cache \
      --quic_cert /tmp/qntc_progressive.crt --quic_key /tmp/qntc_progressive.key \
      --frames 300 --autoplay --video_fps 30 2>&1 | tee /tmp/qntc_dynamic_priority_receiver.log
