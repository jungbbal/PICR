# Latent figure (`figures/fig_latent.{png,pdf}`)

PICR's context z, four panels:

- **(a)** PCA of z over every carry window of `official_train` val (the PCA set) plus `official_ood_2to4kg`, coloured by
  payload mass (trained 0.1-2 kg, unseen 2-4 kg). PC1/PC2 signs are set so both correlate positively with mass. The
  arrow is the mass direction (parallel to z_0.1kg -> z_3kg); the two marked points are the z's of panel (b).
- **(b)** one 3 kg carry window (`official_ood_2to4kg` demo_1195, window 38) rolled out 12 steps (0.24 s) open loop
  with the recorded torques, once with z_0.1kg (the mean z of 20 `official_carry` val episodes at 0.1 kg) and once
  with the window's own z_3kg. Drawn over the Isaac Sim replay of the arm; the number under each panel is the cube's
  end-point error.
- **(c)** carry: PC1 of z against time from the lift, median over episodes at 0.1 / 1.19 / 2 kg. The windows start
  0.5 s before the lift (derived set, see `make_prelift.py`), so all payloads start from the same reading.
- **(d)** the payload swept 0.1 -> 4 kg and 4 -> 0.1 kg during a figure-eight (`ramp8_up`, `ramp8_down`): PC1 follows
  it in real time (median and IQR over episodes).

## Data
Checkpoint `checkpoints/picr.pt`. Datasets (`$PICR_DATA_ROOT/raw/<id>/<id>.hdf5`, contracts in `data/contracts/`):
`official_train`, `official_carry`, `official_ood_2to4kg`, `track8ppo3d_carry`, `track8ppo3d_ood_2to4kg`, `ramp8_up`,
`ramp8_down`. The two figure-eight sets `track8ppo3d_*` are used only for the Spearman numbers, not drawn.
`official_carry_prelift` is derived from `official_carry` by `make_prelift.py` inside the run (written to the work
dir, deleted after encoding). The arm images of (b) are shipped in `assets/frames/` (2.5 MB; see `render/`).

## Commands
```bash
bash paper/fig_latent/run.sh               # intermediates (existing ones are kept) + figure + numbers
bash paper/fig_latent/run.sh --plot-only   # figure + numbers from the intermediates only (no datasets, no GPU)
```
Intermediates go to `${PICR_WORK:-work}/fig_latent/` (~70 MB): `z/<set>.npz` (encode_z.py), `z/pca.npz`
(pca_frame.py), `z/ramp8_*_payload.npy` (sweep_payload.py), `prelift/` (contract + lift frames), `stale_z.json`
(stale_z.py), `arm_window_h12.npz` (arm_window.py), `fig_numbers.json` (plot.py). Delete a file to recompute it.

Outputs: `figures/fig_latent.{png,pdf}` and `results/fig_latent_numbers.txt` (PCA explained variance, Spearman rho of
PC1 vs mass on five sets, the (b) end-point errors, the unseen-3 kg adapted-vs-stale rollout error over 150 episodes,
and (c)'s pre-lift vs carry distance to the own-mass centroid and mass read-out error).

Runtime: about 1-2 min on one GPU for the intermediates (most of it reading the datasets), ~30 s for the figure.

## Files
| file | role |
|---|---|
| `run.sh` | entry point |
| `encode_z.py` | z of every carry window of a contract split |
| `make_prelift.py` | derived pre-lift copy of 200 `official_carry` val episodes + lift frames |
| `sweep_payload.py` | payload mass per window of the sweep sets |
| `pca_frame.py` | PCA frame + sign rule |
| `stale_z.py` | z_0.1kg and the unseen-3 kg rollout check (fixed random episode draws) |
| `arm_window.py` | panel (b)'s rollouts (joint angles + cube poses) |
| `arm_frames.py` | composes the arm panels from the rendered frames |
| `plot.py` | the figure (+ `fig_numbers.json`) |
| `summary_numbers.py` | `results/fig_latent_numbers.txt` |
| `render/` | optional, needs Isaac Lab: re-render the arm frames |

## Re-rendering the arm frames (optional, Isaac Lab)
```bash
python3 render/replay_prep.py --npz work/fig_latent/arm_window_h12.npz --out <replay dir>   # gt/ours/stale.npz
# inside the Isaac Lab container, from the folder holding replay_arm.py:
python.sh replay_arm.py --headless --enable_cameras --traj <replay dir>/gt.npz <replay dir>/ours.npz \
  <replay dir>/stale.npz --out <replay dir>/frames --focal 26 --eye 0.056 -1.881 0.657 --tgt 0.33 -0.07 0.4 --drop_ground
python3 render/pack_frames.py <replay dir>/frames assets/frames --last 12
```
A new render differs from the shipped frames only by renderer noise in the RGB images (depth is identical), i.e.
small pixel differences inside the two (b) arm images.
