#!/bin/bash
# Train PICR on official_train (40 epochs).   bash scripts/train.sh [extra trainer args, e.g. --seed 1 --run_name picr_s1]
# -> $PICR_RUNS_ROOT/<run_name>/last.pt
set -eu
source "$(dirname "$0")/../env.sh"; cd "$PICR_ROOT"
python3 -u scripts/train.py --config configs/picr.yaml --output_dir "$PICR_RUNS_ROOT" "$@"
