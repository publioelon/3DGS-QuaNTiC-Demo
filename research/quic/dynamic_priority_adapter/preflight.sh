#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/3DGS-QuaNTiC-Demo"
python "$DIR/send_dynamic_priority.py" --start 0 --end "${QNTC_END:-29}" --preflight-only
