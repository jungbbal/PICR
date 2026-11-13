#!/usr/bin/env python3
"""Table 1 (tab:causal-results), PICR column, from the three metric files the pipelines write:

    results/table1/add.json    <- paper/fig_payload_sweep/run.sh   (ADD, figure-eight payload sweep)
    results/table1/gain.json   <- paper/fig_counterfactual/run.sh  (do(tau) acceleration gain)
    results/table1/rmse.json   <- paper/fig_lagrangian/run.sh      (joint-2 Lagrangian term RMSE)

Writes results/table1/table1.tex (LaTeX) and prints a Markdown copy. The baseline columns of the paper are not
reproduced by this repository.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PICR_ROOT", Path(__file__).resolve().parents[2]))
RES = Path(os.environ.get("PICR_RESULTS_ROOT", ROOT / "results")) / "table1"
ROWS = ["picr"]
LABELS = ["PICR"]


def load(name):
    return json.load(open(RES / f"{name}.json"))


def main() -> None:
    add, gain, rmse = load("add"), load("gain"), load("rmse")
    # (metric, setting, values per row)
    lines = [
        ("ADD (cm) $\\downarrow$", "Seen (0.1--2\\,kg)", [add["rows"][r]["seen_0.1-2kg"] for r in ROWS], "min"),
        ("", "Unseen (2--4\\,kg)", [add["rows"][r]["unseen_2-4kg"] for r in ROWS], "min"),
        ("Gain $\\rightarrow 1$", "Joints 1--3 (mean)", [gain["rows"][r]["gain"] for r in ROWS], "one"),
        ("RMSE (N\\,m) $\\downarrow$", "Gravity $g_2$", [rmse["rows_full_precision"][r]["gravity"] for r in ROWS], "min"),
        ("", "Inertial $[H\\ddot{q}]_2$", [rmse["rows_full_precision"][r]["inertial"] for r in ROWS], "min"),
        ("", "Coriolis $c_2$", [rmse["rows_full_precision"][r]["coriolis"] for r in ROWS], "min"),
    ]

    def cells(vals, _best):
        return [f"{v:.2f}" for v in vals]

    tex = ["\\begin{tabular}{@{}llc@{}}", "\\toprule",
           "Metric & Setting & " + " & ".join(LABELS) + " \\\\", "\\midrule"]
    md = ["| Metric | Setting | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(ROWS)]
    for i, (metric, setting, vals, best) in enumerate(lines):
        if i in (2, 3):
            tex.append("\\midrule")
        c = cells(vals, best)
        tex.append(f"{metric} & {setting} & " + " & ".join(c) + " \\\\")
        md.append(f"| {metric} | {setting} | " + " | ".join(c) + " |")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (RES / "table1.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\n-> {RES / 'table1.tex'}")


if __name__ == "__main__":
    main()
