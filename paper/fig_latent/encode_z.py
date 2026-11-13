"""Encode the context z of every carry-phase window of a contract split.

    python3 encode_z.py --ckpt checkpoints/picr.pt --contract data/contracts/official_carry.json --out z.npz
        [--split val] [--max-eps N] [--dataset-file <hdf5>]

-> npz with z (N, latent_dim), mass, com, episode, window_start (one row per window).
--max-eps N keeps N episodes of the split, drawn with np.random.default_rng(--seed) and kept in contract order.
--dataset-file overrides $PICR_DATA_ROOT/raw/<dataset_id>/<dataset_id>.hdf5 (used for the derived pre-lift set).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cdelan import paths
from cdelan.analysis._common import build_dataset, load_model
from cdelan.analysis.z_pca_mass import encode, episode_meta

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--contract", required=True)
p.add_argument("--split", default="val")
p.add_argument("--max-eps", type=int, default=0)     # 0 = all
p.add_argument("--seed", type=int, default=0)
p.add_argument("--chunk", type=int, default=60)
p.add_argument("--dataset-file", default=None)
p.add_argument("--out", required=True)
a = p.parse_args()

model, cfg = load_model(a.ckpt)
contract = json.loads(Path(a.contract).read_text())
path = a.dataset_file or str(paths.dataset_file(contract["dataset_id"]))
names = list(contract["splits"][a.split])
if a.max_eps and len(names) > a.max_eps:
    rng = np.random.default_rng(a.seed)
    names = list(np.array(names)[np.sort(rng.choice(len(names), a.max_eps, replace=False))])
meta = episode_meta(path, names)
print(f"{contract['dataset_id']}/{a.split}: {len(names)} episodes, latent_dim={cfg['latent_dim']}", flush=True)

Z, M, COM, EP, WS = [], [], [], [], []
for i in range(0, len(names), a.chunk):
    sub = names[i : i + a.chunk]
    ds = build_dataset([(path, n) for n in sub], cfg)
    Z.append(encode(model, ds.history_states, ds.history_torques))
    nm = np.array(sub)[np.asarray(ds.episode_id)]
    M.append(np.array([meta[n]["mass"] for n in nm], dtype=np.float32))
    COM.append(np.array([meta[n]["com"] for n in nm], dtype=np.float32))
    EP.append(nm)
    WS.append(np.asarray(ds.window_start))
    print(f"  {i + len(sub)}/{len(names)} eps -> {sum(len(x) for x in Z)} windows", flush=True)

out = dict(z=np.concatenate(Z), mass=np.concatenate(M), com=np.concatenate(COM), episode=np.concatenate(EP),
           window_start=np.concatenate(WS))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **out)
print("saved", a.out, {k: v.shape for k, v in out.items()})
