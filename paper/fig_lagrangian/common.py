"""Paths and shared helpers of the Lagrangian-term evaluation (fig_lagrangian, Table 1 RMSE).

Intermediates live in ${PICR_WORK:-$PICR_ROOT/work}/fig_lagrangian/:
    decomp_<row>_ood/torque_decomp_windows.npz   (a) teacher-forced term decomposition on official_ood_2to4kg
    hgmass_<row>/hg_vs_mass.npz                  (c) H, g on a fixed pose bank vs payload mass
    switch012_<row>.npz                          (b) H, g, c on a fixed state through a 0.1 -> 2 kg payload switch
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PICR_ROOT = Path(os.environ.get("PICR_ROOT", HERE.parents[1]))
DATA_ROOT = Path(os.environ.get("PICR_DATA_ROOT", PICR_ROOT / "data"))
CONTRACTS = Path(os.environ.get("PICR_CONTRACTS", PICR_ROOT / "data" / "contracts"))
RESULTS = Path(os.environ.get("PICR_RESULTS_ROOT", PICR_ROOT / "results"))
WORK = Path(os.environ.get("PICR_WORK", PICR_ROOT / "work")) / "fig_lagrangian"

OOD_SET = "official_ood_2to4kg"          # 2-4 kg carry, with the 40-substep dynamics truth (substep_dynamics/)
CARRY_SET = "official_carry"             # 0.1-2 kg carry
SWITCH_SET = "ppo_lat8orig_meantau_switch012_twin_lead_nodrop"   # mid-carry payload switch 0.1 -> 2 kg
SWITCH_M_OLD, SWITCH_M_NEW = 0.1, 2.0

ROWS = [("PICR", "picr")]


def h5_path(dsid: str) -> Path:
    return DATA_ROOT / "raw" / dsid / f"{dsid}.hdf5"


def contract_path(dsid: str) -> Path:
    return CONTRACTS / f"{dsid}.json"


def val_names(dsid: str) -> list[str]:
    names = json.load(open(contract_path(dsid)))["splits"]["val"]
    return [n if isinstance(n, str) else n[-1] for n in names]


def ckpt_path(row: str) -> Path:
    return PICR_ROOT / "checkpoints" / f"{row}.pt"


def rel_to_root(p) -> str:
    """Path for metadata: relative to PICR_ROOT when inside it (no host paths in outputs)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PICR_ROOT.resolve()))
    except ValueError:
        return p.name


def decomp_dir(row: str) -> Path:
    return WORK / f"decomp_{row}_ood"


def hgmass_dir(row: str) -> Path:
    return WORK / f"hgmass_{row}"


def switch_npz(row: str) -> Path:
    return WORK / f"switch012_{row}.npz"


def load_decomp(row: str) -> dict:
    d = decomp_dir(row)
    z = np.load(d / "torque_decomp_windows.npz")
    dt = float(json.load(open(d / "torque_decomp_meta.json"))["dt"])
    key = np.array([f"{e}:{int(r)}" for e, r in zip(z["episode"], z["window_row"])])
    return {**{k: z[k] for k in z.files}, "key": key, "dt": dt}


def load_hgmass(row: str):
    return np.load(hgmass_dir(row) / "hg_vs_mass.npz")


def align(runs):
    """Keep the windows every run scored, in one fixed order, and verify the references agree."""
    common = set(runs[0]["key"])
    for r in runs[1:]:
        common &= set(r["key"])
    order = sorted(common, key=lambda s: (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])))
    out = []
    for r in runs:
        pos = {k: i for i, k in enumerate(r["key"])}
        sel = np.array([pos[k] for k in order])
        out.append({k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(r["key"]) else v)
                    for k, v in r.items()})
    for r in out[1:]:
        assert np.allclose(r["ref_T_sum"], out[0]["ref_T_sum"], atol=1e-6), "simulator reference differs between rows"
        assert np.allclose(r["tau_mean"], out[0]["tau_mean"])
    return out, len(common)


def pick_heavy_episode(r, mass: float):
    """Episode of the payload cell nearest `mass` with the most scored windows."""
    cells = np.unique(r["mass"])
    mass = cells[np.argmin(np.abs(cells - mass))]
    eps = np.unique(r["episode"][np.isclose(r["mass"], mass)])
    return max(eps, key=lambda e: int((r["episode"] == e).sum()))
