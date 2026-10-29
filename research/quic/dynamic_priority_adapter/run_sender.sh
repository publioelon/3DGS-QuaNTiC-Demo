#!/usr/bin/env bash
set -euo pipefail
source "$HOME/venvs/qntcstream/bin/activate"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/3DGS-QuaNTiC-Demo" # allow import of project's existing quic_transport
python -u "$DIR/send_dynamic_priority.py" \
  --start 0 --end "${QNTC_END:-29}" \
  --stage-delay "${QNTC_STAGE_DELAY:-0}" \
  --warmup-delay "${QNTC_WARMUP_DELAY:-2}" "$@" \
  2>&1 | tee /tmp/qntc_dynamic_priority_sender.log
