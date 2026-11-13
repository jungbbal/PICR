"""z_0.1kg ("stale" z) and the unseen-3 kg rollout check of the latent figure.

z_stale = mean encoded z over 20 official_carry val episodes at the lightest trained payload (0.1 kg). It is the z
panel (b) rolls the 3 kg window out with, next to the window's own (adapted) z. The script then rolls out up to 150
official_ood_2to4kg episodes at 3.0 +- 0.1 kg (every carry window, the checkpoint's horizon, recorded torques, open
loop) with the adapted z and with z_stale, and reports the mean final-step cube position error of both.

Episodes are drawn from one np.random.default_rng(0) stream, in this order: 20 official_train val episodes per mass
level (a lattice check; the draw only advances the stream), 20 official_carry episodes for z_stale, 150 reference
episodes at the heaviest trained mass, 150 episodes at 3 kg.

    python3 stale_z.py --ckpt checkpoints/picr.pt --out <work>/stale_z.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from cdelan import paths
from cdelan.analysis._common import DEVICE, build_dataset, load_model
from cdelan.analysis.z_id_ood import randmass_meta
from cdelan.analysis.z_rollout import roll_with_z
from cdelan.analysis.z_pca_mass import encode, episode_meta

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ood-dataset", default="official_ood_2to4kg")
ap.add_argument("--ood-mass", type=float, default=3.0)
ap.add_argument("--ood-tol", type=float, default=0.1)
ap.add_argument("--stale-mass", type=float, default=0.1)
ap.add_argument("--ood-episodes", type=int, default=150)
ap.add_argument("--check-episodes", type=int, default=20)
args = ap.parse_args()
C = Path(os.environ["PICR_CONTRACTS"])

rng = np.random.default_rng(0)
model, cfg = load_model(args.ckpt)
contract = json.loads((C / "official_train.json").read_text())
lat_path = str(paths.dataset_file(contract["dataset_id"]))
ood_path = str(paths.dataset_file(args.ood_dataset))
meta = episode_meta(lat_path, contract["splits"]["val"])
levels = np.array(sorted({round(m["mass"], 3) for m in meta.values()}))

# 1. the lattice-check draw (only advances the random stream)
for lv in levels:
    pool = sorted(n for n, m in meta.items() if round(m["mass"], 3) == lv)
    rng.choice(len(pool), size=min(args.check_episodes, len(pool)), replace=False)

# 2. z_stale: mean z over official_carry val episodes at the lightest level
stale_lv = float(levels[np.argmin(np.abs(levels - args.stale_mass))])
pc = json.loads((C / "official_carry.json").read_text())
id_path = str(paths.dataset_file(pc["dataset_id"]))
pool = sorted(n for n, m in episode_meta(id_path, pc["splits"]["val"]).items() if round(m["mass"], 3) == stale_lv)
names_st = [pool[i] for i in rng.choice(len(pool), size=min(args.check_episodes, len(pool)), replace=False)]
ds_st = build_dataset([(id_path, n) for n in names_st], cfg)
z_stale = encode(model, ds_st.history_states, ds_st.history_torques).mean(0)
print(f"stale z = mean over {len(names_st)} {pc['dataset_id']} val episodes at {stale_lv} kg")

# 3. the reference draw (only advances the random stream), then the unseen-payload episodes
rmeta = randmass_meta(ood_path)
ref = [n for n, m in rmeta.items() if abs(m["mass"] - levels[-1]) <= args.ood_tol and m["n_phase3"] >= 30]
if not ref:
    ref = [n for n, m in meta.items() if round(m["mass"], 3) == levels[-1]]
rng.choice(len(ref), size=min(args.ood_episodes, len(ref)), replace=False)
pool = [n for n, m in rmeta.items() if abs(m["mass"] - args.ood_mass) <= args.ood_tol and m["n_phase3"] >= 30]
names = [pool[i] for i in rng.choice(len(pool), size=min(args.ood_episodes, len(pool)), replace=False)]
ds = build_dataset([(ood_path, n) for n in names], cfg)

r = model.robot_dof
err = {"adapted": [], "stale": []}
for i in range(0, len(ds.history_states), 2048):
    hs = torch.from_numpy(np.asarray(ds.history_states[i : i + 2048])).to(DEVICE)
    ht = torch.from_numpy(np.asarray(ds.history_torques[i : i + 2048])).to(DEVICE)
    ft = torch.from_numpy(np.asarray(ds.future_torques[i : i + 2048])).to(DEVICE)
    gt = np.asarray(ds.future_states[i : i + 2048])[:, :, 2 * r : 2 * r + 3]
    with torch.no_grad():
        z = model.encode_context(hs, ht, deterministic_context=True).mu
    zs = torch.as_tensor(z_stale, dtype=z.dtype, device=DEVICE).expand_as(z)
    for key, zz in (("adapted", z), ("stale", zs)):
        obj = roll_with_z(model, hs, ft, zz).cpu().numpy()
        err[key].append(np.linalg.norm(obj[:, -1, :3] - gt[:, -1], axis=-1) * 100.0)
err = {k: np.concatenate(v) for k, v in err.items()}
H = int(cfg["rollout_horizon"])
print(f"ood {len(names)} eps ~{args.ood_mass} kg, {len(err['adapted'])} windows, h={H} mean final-step cube error: "
      f"adapted {err['adapted'].mean():.2f}  stale {err['stale'].mean():.2f} cm")
Path(args.out).write_text(json.dumps({
    "z_stale": z_stale.tolist(), "stale_level_kg": stale_lv, "stale_episodes": names_st,
    "ood": {"dataset": args.ood_dataset, "mass_kg": args.ood_mass, "n_episodes": len(names),
            "n_windows": int(len(err["adapted"])), "horizon": H,
            "mean_cm_adapted": float(err["adapted"].mean()), "mean_cm_stale": float(err["stale"].mean()),
            "median_cm_adapted": float(np.median(err["adapted"])), "median_cm_stale": float(np.median(err["stale"]))},
}, indent=1))
print("->", args.out)
