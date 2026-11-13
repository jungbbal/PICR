#!/bin/bash
# The paper's latent figure: figures/fig_latent.{png,pdf} + results/fig_latent_numbers.txt.
#   bash paper/fig_latent/run.sh               # intermediates (skips the ones that exist) + figure + numbers
#   bash paper/fig_latent/run.sh --plot-only   # figure + numbers from the existing intermediates (no datasets needed)
# env: CKPT (default checkpoints/picr.pt), PICR_WORK (intermediates go to $PICR_WORK/fig_latent, default work/)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../env.sh"
cd "$PICR_ROOT"
export PYTHONPATH="$HERE:$PYTHONPATH" HDF5_USE_FILE_LOCKING=FALSE
CK=${CKPT:-checkpoints/picr.pt}
W="${PICR_WORK:-$PICR_ROOT/work}/fig_latent"
OUT_FIG="$PICR_ROOT/figures/fig_latent"
OUT_NUM="$PICR_RESULTS_ROOT/fig_latent_numbers.txt"
mkdir -p "$W/z" "$W/prelift" "$(dirname "$OUT_FIG")" "$PICR_RESULTS_ROOT"
log() { echo "$(date '+%H:%M:%S') $*"; }

if [ "${1:-}" != "--plot-only" ]; then
  T0=$(date +%s)
  # 1. z of every carry window of each set (all val episodes)
  for s in official_train official_carry official_ood_2to4kg track8ppo3d_carry track8ppo3d_ood_2to4kg ramp8_up ramp8_down; do
    [ -s "$W/z/$s.npz" ] && continue
    python3 "$HERE/encode_z.py" --ckpt "$CK" --contract "$PICR_CONTRACTS/$s.json" --out "$W/z/$s.npz" > "$W/z/$s.log" 2>&1
    log "encoded $s: $(tail -1 "$W/z/$s.log" | cut -c1-80)"
  done
  # 2. panel (c)'s pre-lift copy of 200 official_carry episodes (derived, built here, deleted after encoding)
  if [ ! -s "$W/z/official_carry_prelift.npz" ]; then
    python3 "$HERE/make_prelift.py" "$W/prelift" 25
    python3 "$HERE/encode_z.py" --ckpt "$CK" --contract "$W/prelift/official_carry_prelift.json" \
      --dataset-file "$W/prelift/official_carry_prelift.hdf5" --out "$W/z/official_carry_prelift.npz" \
      > "$W/z/official_carry_prelift.log" 2>&1
    rm -f "$W/prelift/official_carry_prelift.hdf5"
    log "encoded official_carry_prelift"
  fi
  # 3. payload mass per window of the two sweeps, PCA frame
  [ -s "$W/z/ramp8_down_payload.npy" ] || python3 "$HERE/sweep_payload.py" "$W/z" ramp8_up ramp8_down
  python3 "$HERE/pca_frame.py" "$W/z"
  # 4. z_0.1kg (stale z) + the unseen-3 kg rollout check
  [ -s "$W/stale_z.json" ] || python3 "$HERE/stale_z.py" --ckpt "$CK" --out "$W/stale_z.json"
  # 5. panel (b)'s window, rolled out with both z's (the Isaac replay of it is shipped in assets/frames, see render/)
  [ -s "$W/arm_window_h12.npz" ] || python3 "$HERE/arm_window.py" --ckpt "$CK" --stale "$W/stale_z.json" \
    --episode demo_1195 --window 38 --horizon 12 --out "$W/arm_window_h12.npz"
  log "intermediates done in $(( $(date +%s) - T0 )) s"
fi

# 6. figure + numbers
python3 "$HERE/plot.py" --work "$W" --frames "$HERE/assets/frames" --out "$OUT_FIG" 2>&1 | grep -v findfont
python3 "$HERE/summary_numbers.py" "$W" > "$OUT_NUM"
cat "$OUT_NUM"
log "-> $OUT_FIG.{png,pdf}, $OUT_NUM"
