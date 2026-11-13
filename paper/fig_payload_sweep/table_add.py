"""Table 1, ADD rows (PICR column): mean h = 15 ADD [cm] on the figure-eight payload sweep (ramp8_up), split by the true payload at the
anchor: seen 0.1 < m <= 2 kg (inside the training range) and unseen 2 < m < 4 kg. The pre-ramp 0.1 kg windows and the
post-ramp 4 kg windows are left out.

    python3 paper/fig_payload_sweep/table_add.py [--cache <dir>] [--out results/table1/add.json]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROWS = [("picr", "PICR")]
BANDS = {"seen_0.1-2kg": lambda m: (m > 0.1001) & (m <= 2.0), "unseen_2-4kg": lambda m: (m > 2.0) & (m < 3.999)}


def main():
    root = Path(os.environ.get("PICR_ROOT", Path(__file__).resolve().parents[2]))
    work = Path(os.environ.get("PICR_WORK", root / "work"))
    res = Path(os.environ.get("PICR_RESULTS_ROOT", root / "results"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(work / "fig_payload_sweep" / "cache"))
    ap.add_argument("--out", default=str(res / "table1" / "add.json"))
    a = ap.parse_args()
    out = {"metric": "mean ADD [cm] at h = 15 (300 ms), open loop", "dataset": "ramp8_up (figure-eight, payload 0.1 -> 4 kg)",
           "bands": {"seen_0.1-2kg": "0.1 < m <= 2 kg", "unseen_2-4kg": "2 < m < 4 kg"}, "n_windows": {}, "rows": {}}
    for row, label in ROWS:
        z = np.load(Path(a.cache) / f"{row}_ramp8up_h15.npz")
        key = list(zip(z["episode"].tolist(), z["t"].tolist()))
        m, add = z["mass_kg"], z["add_cm"][:, -1]
        out["rows"][row] = {"label": label}
        for band, sel in BANDS.items():
            s = sel(m)
            out["n_windows"][band] = int(s.sum())
            out["rows"][row][band] = round(float(np.nanmean(add[s])), 4)
        out["n_anchors"] = len(key)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"ADD h15 [cm], ramp8_up, {out['n_anchors']} anchors; windows "
          + ", ".join(f"{b} {n:,}" for b, n in out["n_windows"].items()))
    for row, label in ROWS:
        r = out["rows"][row]
        print(f"   {label:10s} seen {r['seen_0.1-2kg']:6.2f}   unseen {r['unseen_2-4kg']:6.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
