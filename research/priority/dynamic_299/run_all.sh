#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
SCRIPT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT"
python rank_ntc_frames.py --start 0 --end 298 --samples-per-tile 128 --probes 2 --emit-stages
python check_schedule.py --start 0 --end 298 --stages
