# PICR

Code and trained model for the PICR paper (L4DC 2027).
Project page: https://jungbbal.github.io/PICR/

PICR predicts the joint dynamics of a robot arm carrying an unknown payload. A context encoder reads a short history of
joint states and torques and produces a latent z. A Lagrangian model then predicts q̈ = H(q, z)⁻¹ (τ − c(q, q̇, z) − g(q, z)),
where the networks for H (through its Cholesky factor), c and g are FiLM-conditioned on z. Training uses three losses:
an acceleration loss, an inverse-dynamics loss, and a contrastive term on z. The contrastive term uses Gaussian
weights over payload mass and centre of mass.

The experiments use a simulated Franka Panda in Isaac Lab, recorded at 20 ms steps with window-mean measured torques.

## Contents

```
configs/picr.yaml      training config
checkpoints/picr.pt    released weights (epoch 40, seed 0; SHA256SUMS)
src/cdelan/            PICR model, data pipeline, trainer, evaluation
scripts/               train.py / train.sh, pred_quality.py / eval.sh
paper/                 PICR's paper results: one folder per experiment, table1/, run_all.sh
figures/               the paper figures
data/                  contracts/ (episode splits, tracked) and README.md (dataset list); datasets go in data/raw/
docs/                  project page (GitHub Pages)
```

Baseline implementations are not part of this repository.

## Setup

```bash
pip install -r requirements.txt          # Python 3.8, CUDA GPU
source env.sh                            # PYTHONPATH, data / runs / results roots
```

Datasets are not included; see [data/README.md](data/README.md) for the ids and where they go. To keep them on
another disk, put `export PICR_DATA_ROOT=/path/to/data` in an untracked `env.local.sh`.

## Training

```bash
bash scripts/train.sh                                # -> runs/picr/last.pt after 40 epochs
bash scripts/train.sh --seed 1 --run_name picr_s1    # extra trainer flags pass through
```

The released checkpoint is the epoch-40 `last.pt` of `configs/picr.yaml` on `official_train`. Its 40 epochs took
about 5.5 h on one RTX 4090 that was shared by three runs.

## Evaluation

```bash
bash scripts/eval.sh                     # h = 15 (300 ms) open-loop prediction on official_carry and official_ood_2to4kg
# -> results/prediction/picr_<set>_h15.json, prints the mean ADD [cm] of the cube and the per-joint angle error
```

## Paper results

```bash
bash paper/run_all.sh                    # from the checkpoint + datasets (GPU), about 15 min
bash paper/run_all.sh --plot-only        # tables / latent figure from the intermediates in work/
```

| folder | output |
|---|---|
| `paper/fig_payload_sweep` | `results/table1/add.json`: ADD on the figure-eight payload sweep, seen / unseen payload |
| `paper/fig_counterfactual` | `results/table1/gain.json`: acceleration gain under torque interventions |
| `paper/fig_lagrangian` | `results/table1/rmse.json`: RMSE of PICR's g₂, [Hq̈]₂, c₂ vs the simulator |
| `paper/fig_latent` | `figures/fig_latent.{png,pdf}`, `results/fig_latent_numbers.txt` |
| `paper/table1` | `results/table1/table1.tex`: the PICR column of Table 1 |

Each folder's README lists the datasets it needs, its steps and its runtime. The other figures in `figures/` are the
paper versions; their PICR data come from the folders above.

PICR column of Table 1 from the released checkpoint:

| Metric | Setting | PICR |
|---|---|---|
| ADD (cm) ↓ | Seen (0.1–2 kg) | 4.58 |
|  | Unseen (2–4 kg) | 6.92 |
| Gain → 1 | Joints 1–3 (mean) | 0.86 |
| RMSE (N m) ↓ | Gravity g₂ | 1.84 |
|  | Inertial [Hq̈]₂ | 1.63 |
|  | Coriolis c₂ | 1.46 |
