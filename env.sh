# source env.sh   (from anywhere)
# Every path resolves relative to this folder; export a variable before sourcing to override it.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PICR_ROOT="$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PICR_DATA_ROOT="${PICR_DATA_ROOT:-$REPO/data}"           # datasets: $PICR_DATA_ROOT/raw/<dataset_id>/<dataset_id>.hdf5
export PICR_RUNS_ROOT="${PICR_RUNS_ROOT:-$REPO/runs}"           # training runs
export PICR_RESULTS_ROOT="${PICR_RESULTS_ROOT:-$REPO/results}"  # evaluation outputs
export PICR_CONTRACTS="${PICR_CONTRACTS:-$REPO/data/contracts}" # episode splits / dataset contracts (tracked)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# machine-specific overrides (untracked), e.g. export PICR_DATA_ROOT=/mnt/data/picr
[ -f "$REPO/env.local.sh" ] && source "$REPO/env.local.sh"
