#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
cd "$(dirname "$0")"
python map_gaussian_blocks.py --ntc 114 --batch-size 2048 --cuda-probes 12 "$@"
