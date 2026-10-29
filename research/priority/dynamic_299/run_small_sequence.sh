#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
SCRIPT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT"
python rank_ntc_frames.py --start 112 --end 116 --samples-per-tile 128 --probes 2 --emit-stages
python check_schedule.py --start 112 --end 116 --stages
python compare_frames.py 114 115
