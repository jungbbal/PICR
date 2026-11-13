"""Table 1 RMSE rows (PICR) + the numbers quoted in the text for the Lagrangian-structure figure.

 (a)  shown episode = the 3 kg episode of panel (a) (pick_heavy_episode on the PICR windows of official_ood_2to4kg), first
      1 s, joint 2: RMSE of each PICR term vs the simulator's 40-substep mean term (gravity g_2, inertial [H qdd]_2,
      Coriolis c_2)  -> $PICR_RESULTS_ROOT/table1/rmse.json  (Table 1 RMSE column)
 (a') the same RMSE over all common OOD windows (2-4 kg)
 (b)  payload switch 0.1 -> 2 kg: median over episodes at each frame relative to the switch; crossing = first frame whose
      median passes 50 % / 90 % of the simulator step (gt_old -> gt_new); pre = median of rel -40..-1, post = median of
      rel 40..80, both as a fraction of the step.
 (a'), (b) -> ${PICR_WORK:-$PICR_ROOT/work}/fig_lagrangian/numbers.md

    python3 paper/fig_lagrangian/table1_rmse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OOD_SET, RESULTS, ROWS, WORK, align, load_decomp, pick_heavy_episode, switch_npz  # noqa: E402

J, T_MAX, MASS_A = 1, 1.0, 3.0
TERMS = (("T_G", "gravity", "gravity g_2"), ("T_M", "inertial", "inertial [H qdd]_2"), ("T_C", "coriolis", "Coriolis c_2"))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    runs, _ = align([load_decomp(row) for _, row in ROWS])
    ep = pick_heavy_episode(runs[0], MASS_A)
    k = runs[0]["episode"] == ep
    m_a = float(runs[0]["mass"][k][0])
    k = k & (np.cumsum(k) * runs[0]["dt"] <= T_MAX + 1e-9)

    table = {row: {key: rmse(r[f"model_{t}"][k, J], r[f"ref_{t}"][k, J]) for t, key, _ in TERMS}
             for (_, row), r in zip(ROWS, runs)}
    out = RESULTS / "table1" / "rmse.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "metric": "RMSE [N m] of joint-2 Lagrangian terms vs the simulator's 40-substep mean terms, teacher-forced",
        "dataset": OOD_SET, "episode": str(ep), "payload_kg": m_a, "window_s": T_MAX, "n_windows": int(k.sum()),
        "joint": J + 1, "terms": {key: lab for _, key, lab in TERMS},
        "rows": {row: {key: round(v, 2) for key, v in vals.items()} for row, vals in table.items()},
        "rows_full_precision": table,
    }, open(out, "w"), indent=1)

    L = ["# Numbers behind fig_lagrangian (panels a, b)", "",
         f"## (a) episode {ep}, {m_a:.3g} kg, first {T_MAX:g} s = {int(k.sum())} windows ({OOD_SET}); RMSE [N m], joint 2", "",
         "| row | " + " | ".join(lab for *_, lab in TERMS) + " |", "|---|---|---|---|"]
    for (n, row) in ROWS:
        L.append(f"| {n} | " + " | ".join(f"{table[row][key]:.2f}" for _, key, _ in TERMS) + " |")
    L += ["", f"## (a') all {len(runs[0]['key'])} common OOD windows (2-4 kg); RMSE [N m], joint 2", "",
          "| row | " + " | ".join(lab for *_, lab in TERMS) + " |", "|---|---|---|---|"]
    for (n, _), r in zip(ROWS, runs):
        L.append(f"| {n} | " + " | ".join(f"{rmse(r[f'model_{t}'][:, J], r[f'ref_{t}'][:, J]):.2f}" for t, *_ in TERMS) + " |")
    L += ["", "## (b) payload switch 0.1 -> 2 kg on the fixed representative state; median over episodes", "",
          "| row | term | simulator old -> new | pre-switch median (frac. of step) | post-switch median (frac.) | 50 % crossing | 90 % crossing |",
          "|---|---|---|---|---|---|---|"]
    for n, row in ROWS:
        d = np.load(switch_npz(row))
        rel = d["rel"]
        for key, ki, unit in (("g2", 3, "N m"), ("H22", 0, "kg m^2")):
            v = d[f"rep_{key}"]
            go, gn = d["rep_gt_old"][ki], d["rep_gt_new"][ki]
            pre = np.median(v[(rel >= -40) & (rel <= -1)]); post = np.median(v[(rel >= 40) & (rel <= 80)])
            fr = {x: (np.median(v[rel == x]) - go) / (gn - go) for x in range(0, 81)}

            def cross(p):
                if (pre - go) / (gn - go) >= p:                      # already beyond that level before the switch
                    return "n/a (pre-switch value already past it)"
                c = next((x for x in fr if fr[x] >= p), None)
                return "not reached (<= 1.6 s)" if c is None else f"frame {c} = {20 * c} ms"
            L.append(f"| {n} | {key} [{unit}] | {go:.3f} -> {gn:.3f} | {pre:.3f} ({(pre - go) / (gn - go):.2f}) | "
                     f"{post:.3f} ({(post - go) / (gn - go):.2f}) | {cross(0.5)} | {cross(0.9)} |")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "numbers.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
