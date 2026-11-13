# Data

Datasets are not part of this repository. Place each one as

    $PICR_DATA_ROOT/raw/<dataset_id>/<dataset_id>.hdf5

(`PICR_DATA_ROOT` defaults to this `data/` folder; override it in `env.local.sh`). The episode splits of every dataset
are tracked in `contracts/<dataset_id>.json`.

Every dataset is a recording of a simulated Franka Panda (Isaac Lab) carrying a cube payload. All of them log the
joint states and the 20 ms window-mean measured joint torques at 50 Hz.

| dataset_id | content | used by |
|---|---|---|
| `official_train` | training set: pick-and-carry, payload 0.1-2 kg, CoM lattice | `scripts/train.sh`, `fig_latent` |
| `official_carry` | test, carry, payload 0.1-2 kg (unseen states) | `scripts/eval.sh`, `fig_lagrangian` (c), `fig_latent` |
| `official_ood_2to4kg` | test, carry, payload 2-4 kg (unseen masses); includes the 40-substep M, c, g truth | `scripts/eval.sh`, `fig_lagrangian` (a)(c), `fig_latent` |
| `ramp8_up` | figure-eight, payload ramped 0.1 -> 4 kg | `fig_payload_sweep` (Table 1 ADD), `fig_latent` |
| `ramp8_down` | figure-eight, payload ramped 4 -> 0.1 kg | `fig_latent` |
| `track8ppo3d_carry` | figure-eight, constant payload 0.1-2 kg | `fig_latent` (numbers) |
| `track8ppo3d_ood_2to4kg` | figure-eight, constant payload 2-4 kg | `fig_latent` (numbers) |
| `ppo_lat8orig_meantau_switch012_twin_lead_nodrop` | mid-carry payload switch 0.1 -> 2 kg | `fig_lagrangian` (b) |
| `official_intervene_jlm7000_h13` | torque interventions do(tau), +dtau branches and paired null runs | `fig_counterfactual` (Table 1 Gain) |
| `official_intervene_jlm7000n_h13` | the same, -dtau branches | `fig_counterfactual` (Table 1 Gain) |

`fig_latent` also builds `official_carry_prelift` from `official_carry` itself (`paper/fig_latent/make_prelift.py`).
