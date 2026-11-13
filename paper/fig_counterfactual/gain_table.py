#!/usr/bin/env python3
"""Projected acceleration gain of PICR on the do(tau) fork pairs (Table 1 "Gain").

For a pushed branch i and its null branch j0 of the same env, the simulator's change of joint acceleration in the first
20 ms is dq = (dq_i - dq_j0) / 0.02 (7 joints) and the model's predicted change is dp (same, from the frozen-context
prediction). gain = <dp, dq> / |dq|^2 (1 = exact). Clean gate: cube slip in the hand <= 1 cm on the branch and no
velocity-limit hit on the null. Both signs (+dtau, -dtau) are pooled; the median is taken per joint (1-3) and per
|dtau| rung (10/20/30/40 N m). Table 1 reports the mean of these 12 medians.

    python3 gain_table.py --dumps work/fig_counterfactual/dumps --out results/table1/gain.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [("picr", "PICR")]
SETS = ("jlm7000", "jlm7000n")
DQ = slice(7, 14)
JR = {"j1": (2, 3, 4, 5), "j2": (6, 7, 8, 9), "j3": (10, 11, 12, 13)}       # branch ids of the 4 rungs per joint
NM = [10, 20, 30, 40]


def gains(D, row, lad):
    """per rung: the gain of every clean anchor (both signs pooled)."""
    G = [[] for _ in NM]
    for t in SETS:
        T = D[("picr", t)]
        P = D[(row, t)]["pred_frozen"]
        key = {(int(b), int(e)): i for i, (b, e) in enumerate(zip(T["branch"], T["env"]))}
        for k in range(len(NM)):
            for (b, e), i in key.items():
                if b != JR[lad][k] or (0, e) not in key:
                    continue
                j0 = key[(0, e)]
                if T["slip_m"][i] > 0.01 or T["wall"][j0]:
                    continue
                dt = (T["fut_states"][i][0, DQ] - T["fut_states"][j0][0, DQ]) / 0.02
                dp = (P[i][0, DQ] - P[j0][0, DQ]) / 0.02
                G[k].append(float(dp @ dt / (dt @ dt)))
    return G


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = {(r, t): dict(np.load(Path(a.dumps) / f"traj_{r}_{t}.npz")) for r, _ in ROWS for t in SETS}

    res = {}
    for r, name in ROWS:
        med, n, iqr = {}, {}, {}
        print(name)
        for lad in JR:
            G = gains(D, r, lad)
            med[lad] = [float(np.median(g)) for g in G]
            n[lad] = [len(g) for g in G]
            q = np.percentile(G[1], [25, 75])
            iqr[lad] = [float(q[0]), float(q[1])]
            print(f"  {lad} gain median 10/20/30/40: " + "/".join(f"{m:.2f}" for m in med[lad])
                  + f"  n {n[lad]}  IQR@20 [{q[0]:.2f},{q[1]:.2f}]")
        g12 = float(np.mean([m for lad in JR for m in med[lad]]))
        print(f"  Gain (mean of 12 medians) {g12:.4f}")
        res[r] = dict(display=name, gain=g12, gain_2dp=round(g12, 2), medians=med, n=n, iqr_20nm=iqr)

    out = dict(metric="gain",
               description="projected acceleration gain <dqdd_pred, dqdd_true>/|dqdd_true|^2 in the first 20 ms after a "
                           "do(tau) push, paired branch vs null; mean of the 12 medians over joints 1-3 x |dtau| "
                           "10/20/30/40 N m (both signs pooled; clean gate: slip <= 1 cm, no velocity-limit hit)",
               datasets=[f"official_intervene_{t}_h13" for t in SETS],
               checkpoints={r: f"checkpoints/{r}.pt" for r, _ in ROWS},
               rows=res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("Gain:", "  ".join(f"{res[r]['display']} {res[r]['gain_2dp']:.2f}" for r, _ in ROWS))
    print("->", a.out)


if __name__ == "__main__":
    main()
