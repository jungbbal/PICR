#!/bin/bash
# h = 15 open-loop prediction of PICR on the evaluation sets.
#   bash scripts/eval.sh [sets...]        sets (default "carry ood"):
#     carry = official_carry (0.1-2 kg), ood = official_ood_2to4kg (2-4 kg), or any dataset id with a contract
#   CKPT=<path> scores another checkpoint (default checkpoints/picr.pt)
# -> $PICR_RESULTS_ROOT/prediction/<ckpt name>_<set>_h15.json (+ .log)
set -eu
source "$(dirname "$0")/../env.sh"; cd "$PICR_ROOT"
sets=${*:-carry ood}
ck=${CKPT:-checkpoints/picr.pt}; tag=$(basename "$ck" .pt); O=$PICR_RESULTS_ROOT/prediction; mkdir -p "$O"
for set in $sets; do
  case $set in carry) name=official_carry ;; ood) name=official_ood_2to4kg ;; *) name=$set ;; esac
  out=$O/${tag}_${set}_h15.json
  python3 scripts/pred_quality.py --ckpt "$ck" --protocol "$PICR_CONTRACTS/$name.json" \
    --dataset_file "$PICR_DATA_ROOT/raw/$name/$name.hdf5" --horizon 15 --out "$out" > "${out%.json}.log" 2>&1
  echo "$tag $set h15: $(grep -m1 'ADD mean' "${out%.json}.log")"
done
