#!/usr/bin/env python3
"""Cut the do(tau) branch recordings down to the one window per episode that the figure scores (optional reference;
the figure reads the finished datasets official_intervene_jlm7000_h13 / official_intervene_jlm7000n_h13).

A recording batch is 14 branches x up to 256 envs, 250 rows each. Every episode is trimmed to rows k-13..k+15 around
its own anchor k (13 pre-anchor rows: a history_torque_lag 1 row needs tau[k-13] for its 12-frame encoder window), the
batches are merged into one file (episodes renumbered demo_0.. in batch order) and the evaluation contract is written.

    python3 build_dataset.py --id official_intervene_jlm7000_h13 --out_root $PICR_DATA_ROOT \
        --contracts $PICR_CONTRACTS <post-processed batch files ..._b00.hdf5 ... _b13.hdf5>
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def trim(src: str, dst: str, history: int, horizon: int) -> None:
    kept = dropped = 0
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        do = fo.create_group("data")
        for k_, v in fi["data"].attrs.items():
            do.attrs[k_] = v
        for name in fi["data"]:
            ep = fi["data"][name]
            k = int(np.asarray(ep["intervention/anchor_step"])[0, 0])
            T = len(np.asarray(ep["obs/joint_pos"]))
            lo, hi = k - history, k + horizon                 # inclusive row range
            if lo < 0 or hi >= T:
                dropped += 1
                continue
            g = do.create_group(name)
            for at, v in ep.attrs.items():
                g.attrs[at] = v

            def recurse(s, d):
                for key, item in s.items():
                    if isinstance(item, h5py.Group):
                        recurse(item, d.create_group(key))
                    else:
                        arr = np.asarray(item)
                        # per-timestep arrays are trimmed; per-episode ones are copied whole
                        d.create_dataset(key, data=arr[lo:hi + 1] if len(arr) == T else arr)

            recurse(ep, g)
            kept += 1
    print(f"  {Path(dst).stem}: {kept} episodes trimmed to {history + horizon + 1} rows, {dropped} dropped")


def merge(dst: str, srcs: list[str]) -> list[str]:
    order, n = [], 0
    with h5py.File(dst, "w") as fout:
        dout = fout.create_group("data")
        for path in srcs:
            with h5py.File(path, "r") as fin:
                for name, value in fin["data"].attrs.items():
                    if name not in dout.attrs:
                        dout.attrs[name] = value
                for ep_name in fin["data"]:
                    fin.copy(fin["data"][ep_name], dout, name=f"demo_{n}")
                    order.append({"demo": f"demo_{n}", "src": os.path.basename(path), "src_episode": ep_name})
                    n += 1
        dout.attrs["merged_from"] = json.dumps(order)
        dout.attrs["num_episodes"] = n
    print(f"wrote {dst}: {n} episodes from {len(srcs)} files")
    return [o["demo"] for o in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="post-processed branch batch files, named ..._bNN.hdf5")
    ap.add_argument("--id", required=True)
    ap.add_argument("--out_root", required=True, help="writes <out_root>/raw/<id>/<id>.hdf5")
    ap.add_argument("--contracts", required=True, help="writes <contracts>/<id>.json")
    ap.add_argument("--history", type=int, default=13)
    ap.add_argument("--horizon", type=int, default=15)
    a = ap.parse_args()
    dst = Path(a.out_root) / "raw" / a.id / f"{a.id}.hdf5"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for s in sorted(a.src, key=lambda p: Path(p).stem.rsplit("_", 1)[-1]):
            part = Path(tmp) / f"{Path(s).stem.rsplit('_', 1)[-1]}.hdf5"        # bNN.hdf5
            trim(s, str(part), a.history, a.horizon)
            parts.append(str(part))
        eps = merge(str(dst), parts)
    sign = "-" if a.id.split("_")[2].endswith("n") else "+"
    contract = {"val": sorted(eps, key=lambda s: int(s.split("_")[1])), "train": [],
                "note": f"torque-intervention branches ({sign}dtau), cut to rows k-{a.history}..k+{a.horizon} around each "
                        "anchor k; pair by intervention/{branch,env_index}"}
    Path(a.contracts).mkdir(parents=True, exist_ok=True)
    json.dump(contract, open(Path(a.contracts) / f"{a.id}.json", "w"), indent=1)
    print(f"{len(eps)} episodes -> {Path(a.contracts) / f'{a.id}.json'}")


if __name__ == "__main__":
    main()
