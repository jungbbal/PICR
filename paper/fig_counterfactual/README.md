# Torque interventions do(tau): Table 1 "Gain" (PICR)

During a carry, a constant torque step is added to one joint at an anchor step k and held for 0.3 s. That is the
intervention do(tau). The same episode without the push is the null branch; every pushed branch has a null branch
from the same env, and the two match exactly up to k. This is the data behind `figures/fig_counterfactual.png`
(PICR rows) and the Table 1 Gain row.

**Gain** = `<dqdd_pred, dqdd_true> / |dqdd_true|^2`, the projected change of joint acceleration in the first 20 ms,
branch minus null (1 = exact). The median is taken over every clean anchor per joint (1-3) and per
|dtau| = 10/20/30/40 N m, both signs pooled; Table 1 reports the mean of these 12 medians. Expected: 0.86.
- Clean gate: the cube slips <= 1 cm in the hand on the branch, and the null branch does not hit the velocity limit.
- PICR is given the context encoded from the null branch's window (`pred_frozen`), so only the torque differs between
  branch and null.

## Datasets
`official_intervene_jlm7000_h13` (+dtau) and `official_intervene_jlm7000n_h13` (-dtau), each at
`$PICR_DATA_ROOT/raw/<id>/<id>.hdf5`, contracts in `data/contracts/<id>.json`.
- Episodes: 3475 and 3469, cut to rows k-13..k+15 around each anchor (3277 and 3276 anchor windows).
- Joints 1-3 x 4 rungs = 12 branches plus the null, over 256 envs.
- The 13th pre-anchor row is needed because PICR's encoder reads tau[k-13..k-2] (`history_torque_lag 1`).
- `build_dataset.py` (optional, reference) is the trimming and merging step that made these files from the
  post-processed Isaac Lab branch recordings.

## Commands (from the repo root)
```
bash paper/fig_counterfactual/run.sh               # dumps (GPU, kept if present) -> results/table1/gain.json
bash paper/fig_counterfactual/run.sh --plot-only   # gain.json from existing dumps
```
`CF_DATA_ROOT=<dir>` reads the two datasets from `<dir>/raw/...` instead of `$PICR_DATA_ROOT`.

Steps:
1. `dump_traj.py`: one dump per set, 10-20 s each on one RTX 4090 ->
   `work/fig_counterfactual/dumps/traj_picr_<jlm7000|jlm7000n>.npz` (about 21 MB each).
2. `gain_table.py`: writes `results/table1/gain.json` and prints the per-joint medians.
