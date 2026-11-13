#!/bin/bash
# PICR's Lagrangian terms vs the simulator: Table 1 RMSE rows + the PICR data of the Lagrangian figure.
#   bash paper/fig_lagrangian/run.sh              # intermediates (skipped if present) -> Table 1 json
#   bash paper/fig_lagrangian/run.sh --plot-only  # Table 1 json from existing intermediates
# Needs (unless --plot-only) $PICR_DATA_ROOT/raw/<id>/<id>.hdf5 for
#   official_ood_2to4kg, official_carry, ppo_lat8orig_meantau_switch012_twin_lead_nodrop
# Intermediates -> ${PICR_WORK:-$PICR_ROOT/work}/fig_lagrangian/ ; output -> results/table1/rmse.json
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
P="$PICR_ROOT/paper/fig_lagrangian"
W="${PICR_WORK:-$PICR_ROOT/work}/fig_lagrangian"
mkdir -p "$W"

if [ "${1:-}" != "--plot-only" ]; then
  for row in picr; do
    ck="$PICR_ROOT/checkpoints/$row.pt"
    if [ ! -f "$W/decomp_${row}_ood/torque_decomp_windows.npz" ]; then
      echo "$(date +%T) (a) torque decomposition, $row"
      python3 -u "$P/torque_decomp.py" --ckpt "$ck" --out_dir "$W/decomp_${row}_ood" > "$W/decomp_${row}_ood.log" 2>&1
    fi
    if [ ! -f "$W/hgmass_${row}/hg_vs_mass.npz" ]; then
      echo "$(date +%T) (c) H, g vs payload mass, $row"
      python3 -u "$P/hg_vs_mass.py" --row "$row" --ckpt "$ck" --out_dir "$W/hgmass_${row}" > "$W/hgmass_${row}.log" 2>&1
    fi
    if [ ! -f "$W/switch012_${row}.npz" ]; then
      echo "$(date +%T) (b) payload switch 0.1 -> 2 kg, $row"
      python3 -u "$P/switch_mcg.py" --ckpt "$ck" --out "$W/switch012_${row}.npz" > "$W/switch012_${row}.log" 2>&1
    fi
  done
fi
echo "$(date +%T) Table 1 RMSE"
python3 "$P/table1_rmse.py" | tail -n 1
echo "$(date +%T) done"
