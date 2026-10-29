#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
SCRIPT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT"
python test_offline.py
python prepare_dynamic_saliency.py \
  --out "$HOME/snap/firefox/common/NTC_dynamic_saliency_timeline.json"
python rank_ntc_frames.py --start 114 --end 114 --samples-per-tile 128 --probes 2 --emit-stages
python check_schedule.py --start 114 --end 114 --stages
