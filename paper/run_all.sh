#!/bin/bash
# PICR's paper results from checkpoints/picr.pt:
#   results/table1/{add,gain,rmse}.json + table1.tex (PICR column), figures/fig_latent.{png,pdf}
#   bash paper/run_all.sh              compute intermediates (GPU + datasets; steps whose outputs exist are skipped)
#   bash paper/run_all.sh --plot-only  tables / latent figure from existing intermediates in ${PICR_WORK:-work}/
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../env.sh"; cd "$PICR_ROOT"
for fig in fig_payload_sweep fig_counterfactual fig_lagrangian fig_latent; do
  echo "=== $fig"
  bash "paper/$fig/run.sh" "$@"
done
echo "=== Table 1"
python3 paper/table1/make_table1.py
