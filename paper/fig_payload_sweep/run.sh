#!/bin/bash
# Table 1 ADD (PICR column) on the figure-eight payload sweep, from checkpoints/picr.pt and the ramp8_up dataset.
#   bash paper/fig_payload_sweep/run.sh              rollout (skipped if the cache exists) -> results/table1/add.json
#   bash paper/fig_payload_sweep/run.sh --plot-only  table from the existing cache
set -euo pipefail
source "$(dirname "$0")/../../env.sh"; cd "$PICR_ROOT"
D=paper/fig_payload_sweep
C="${PICR_WORK:-$PICR_ROOT/work}/fig_payload_sweep/cache"; LOG="${PICR_WORK:-$PICR_ROOT/work}/fig_payload_sweep/logs"
mkdir -p "$C" "$LOG"
if [ "${1:-}" != "--plot-only" ]; then
  if [ -f "$C/picr_ramp8up_h15.npz" ]; then echo "rollout picr: cache exists, skipped"
  else
    python3 $D/rollouts.py --out "$C" > "$LOG/roll_picr.log" 2>&1 || { tail -5 "$LOG/roll_picr.log"; exit 1; }
    echo "rollout picr: $(grep -m1 'ADD mean' "$LOG/roll_picr.log")"
  fi
fi
python3 $D/table_add.py --cache "$C"
