# Payload sweep: Table 1 ADD (PICR)

Open-loop cube prediction on the figure-eight set whose payload ramps from 0.1 to 4 kg over 20 s (`ramp8_up`,
33 episodes, evaluation only). This is the data behind `figures/fig_payload_sweep.png` (PICR column) and the Table 1
ADD rows.

`results/table1/add.json`: mean h = 15 (300 ms) ADD [cm] of PICR, split by the true payload at the anchor.
Seen = 0.1 < m <= 2 kg (16,071 windows), unseen = 2 < m < 4 kg (16,896 windows). Anchors are the 50,848 windows whose
rows t-12..t all lie in the carry phase. Expected: seen 4.58, unseen 6.92.

## Data

`$PICR_DATA_ROOT/raw/ramp8_up/ramp8_up.hdf5`, split `data/contracts/ramp8_up.json` (all 33 episodes in `val`).

## Commands

    bash paper/fig_payload_sweep/run.sh              # rollout (skipped if the cache exists) -> table
    bash paper/fig_payload_sweep/run.sh --plot-only  # table from the existing cache (CPU only)

Single steps: `rollouts.py [--ckpt checkpoints/picr.pt]`, `table_add.py`.

The cache `${PICR_WORK:-work}/fig_payload_sweep/cache/picr_ramp8up_h15.npz` (about 42 MB) holds per-anchor episode, t,
add_cm (N x 15), true / predicted cube pose, the cube position at the anchor and the true payload.

Runtime: about 1.5 min on one RTX 4090.
