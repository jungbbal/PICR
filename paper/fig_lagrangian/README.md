# Lagrangian terms: Table 1 RMSE (PICR)

PICR's learned terms are read against the simulator on joint 2. This is the data behind `figures/fig_lagrangian.png`
(PICR rows) and the Table 1 RMSE rows.

- **(a) separation**: one 3 kg carry episode (unseen mass; `demo_2130` of `official_ood_2to4kg`), first 1 s. Gravity
  g_2, inertial [H qdd]_2 and Coriolis c_2 vs the simulator's 40-substep mean terms (Isaac's articulation terms + the
  analytic rigid-grasp payload terms), teacher-forced at the logged states with the true window-mean acceleration.
  The joint-2 RMSE on this episode is the Table 1 RMSE column. Expected: 1.84 / 1.63 / 1.46 N m.
- **(b) mass change**: mid-carry payload switch 0.1 -> 2 kg. g_2 and H_22 on one fixed state, evaluated at each
  episode's encoded z_t (472 episodes); the 50 % / 90 % crossing times are written to `numbers.md`.
- **(c) unseen mass**: g_2 and H_22 on a fixed bank of 1500 poses vs payload mass 0.1-3 kg, at the mean encoded z of
  each mass cell, next to the simulator's (exactly linear) curve.

A frame is kept only if rows t-13..t are in the carry phase; PICR reads its history torques one row late
(`history_torque_lag 1`, see `lagterms.py`).

## Datasets

`$PICR_DATA_ROOT/raw/<id>/<id>.hdf5` (episode lists in `data/contracts/<id>.json`):

| id | used by |
|---|---|
| `official_ood_2to4kg` | (a), (c). Must contain the `substep_dynamics/` group (40-substep M, c, g truth) |
| `official_carry` | (c) |
| `ppo_lat8orig_meantau_switch012_twin_lead_nodrop` | (b) |

## Commands (from the repo root)

    bash paper/fig_lagrangian/run.sh              # intermediates (skipped if present) -> results/table1/rmse.json
    bash paper/fig_lagrangian/run.sh --plot-only  # rmse.json from existing intermediates (seconds, no GPU)

Steps: `torque_decomp.py` (a), `hg_vs_mass.py` (c), `switch_mcg.py` (b), `table1_rmse.py` (Table 1 json +
`work/fig_lagrangian/numbers.md`). Each has `--help`.

Runtime: about 10 min on one RTX 4090 ((a) about 7.5 min, (b) about 2 min, (c) about 20 s).
