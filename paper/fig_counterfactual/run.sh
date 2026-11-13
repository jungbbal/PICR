#!/usr/bin/env bash
# Torque interventions do(tau): Table 1 Gain (PICR) from checkpoints/picr.pt.
#   bash paper/fig_counterfactual/run.sh              dumps (GPU, skipped when present) -> results/table1/gain.json
#   bash paper/fig_counterfactual/run.sh --plot-only  gain.json from existing dumps
# Optional: CF_DATA_ROOT=<dir> reads the two do(tau) datasets from <dir>/raw/<id>/<id>.hdf5 instead of $PICR_DATA_ROOT.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../env.sh"
cd "$PICR_ROOT"
W="${PICR_WORK:-$PICR_ROOT/work}/fig_counterfactual"
DATA_ROOT="${CF_DATA_ROOT:-$PICR_DATA_ROOT}"
ROWS="picr"
SETS="jlm7000 jlm7000n"                       # +dtau / -dtau branches of the same anchors
mkdir -p "$W/dumps" "$W/logs"
t0=$(date +%s)

if [ "${1:-}" != "--plot-only" ]; then
  # 1. PICR's predictions on every fork pair (about 20 s per set on one GPU)
  for row in $ROWS; do
    for s in $SETS; do
      out="$W/dumps/traj_${row}_$s.npz"
      if [ -s "$out" ]; then echo "[dump] keep $out"; continue; fi
      python3 "$HERE/dump_traj.py" --row "$row" --ckpt "checkpoints/$row.pt" --dataset "official_intervene_${s}_h13" \
        --data_root "$DATA_ROOT" --out "$out" > "$W/logs/dump_${row}_$s.log" 2>&1
      echo "[dump] $row $s: $(grep -m1 'anchor windows' "$W/logs/dump_${row}_$s.log")"
    done
  done
fi
# 2. Table 1 Gain: mean of the 12 medians (joints 1-3 x |dtau| 10/20/30/40 N m)
python3 "$HERE/gain_table.py" --dumps "$W/dumps" --out "$PICR_RESULTS_ROOT/table1/gain.json" | tee "$W/logs/gain.txt"

echo "[run] done in $(( $(date +%s) - t0 )) s"
