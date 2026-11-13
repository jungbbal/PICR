"""Panel (c): does the model's H and g GROW with the payload the way the simulator's do?

A heavier payload makes H and g larger, in a specific q-dependent way, and the only channel a context-conditioned
Lagrangian model has for that is z. So:

    GT      H_full(q, m) = H_arm(q) + m * A(q),     g_full(q, m) = g_arm(q) + m * a(q)
            A, a = the rigid-grasp payload terms of mcg_vs_gt.payload_terms per kg. Exactly linear in m because every
            test set holds the SAME cube geometry (I/m fixed, CoM at the link origin), so scaling a frame's logged
            (mass, inertia) to a counterfactual m scales dH and dg by the same factor and nothing else.
    model   H(q, z_bar(m)), g(q, z_bar(m))
            z_bar(m) = the MEAN ENCODED z of the windows whose true payload is m -- the model's own read of that mass
            from 12 frames of history, not a line fitted through z.

Both are read on ONE fixed pose bank (frames sampled across every mass cell of official_carry and official_ood_2to4kg),
so the curve against m moves for one reason only: the payload. Two blocks land in the npz:

  sweep_*   the bank, swept: per mass cell the bank-mean H (7x7) and g (7) of the model at z_bar(m) and of GT at that m.
  real_*    the same frames' OWN mass, OWN z: the pose-matched mean error mean(H_model - H_full), mean(g_model - g_full)
            per cell, i.e. the bias at the mass the encoder actually saw.

    python3 hg_vs_mass.py --row picr --ckpt checkpoints/picr.pt --out_dir work/fig_lagrangian/hgmass_picr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CARRY_SET, OOD_SET, h5_path, rel_to_root, val_names   # noqa: E402
from lagterms import episode_terms, report                               # noqa: E402
from cdelan.analysis.mcg_vs_gt import DEVICE, N_ARM                      # noqa: E402
from cdelan.eval.rollout import load_checkpoint_model                    # noqa: E402

SETS = {"carry": CARRY_SET, "ood": OOD_SET}
KEEP = ("mass", "z", "H_model", "g_model", "H_arm", "g_arm", "H_pay", "g_pay", "q_abs", "dq")


def episode_masses(h5: str, names: list[str]) -> np.ndarray:
    with h5py.File(h5, "r") as f:
        return np.array([float(np.asarray(f["data"][n]["object_dynamics/mass"]).reshape(-1)[0]) for n in names])


def collect(model, cfg, eps_per_cell: int) -> dict:
    """Per-frame terms over `eps_per_cell` episodes of every payload cell of both sets."""
    out = {k: [] for k in KEEP}
    for tag, dsid in SETS.items():
        h5 = str(h5_path(dsid))
        names = val_names(dsid)
        ms = episode_masses(h5, names)
        for cell in np.unique(np.round(ms, 3)):
            pick = [names[i] for i in np.where(np.round(ms, 3) == cell)[0][:eps_per_cell]]
            got = 0
            for name in pick:
                r = episode_terms(model, cfg, h5, name)
                if r is None:
                    continue
                for k in KEEP:
                    out[k].append(r[k])
                got += len(r["mass"])
            print(f"{tag} {cell:5.3f} kg: {len(pick)} episodes, {got} frames", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--row", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eps_per_cell", type=int, default=8)
    ap.add_argument("--bank", type=int, default=1500, help="poses the sweep is read on")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_checkpoint_model(args.ckpt, device=DEVICE)
    model, cfg = loaded.model.eval(), dict(loaded.config.__dict__)
    assert int(cfg["robot_dof"]) == N_ARM
    has_z = bool(getattr(model, "use_context_encoder", True)) and bool(getattr(model, "delan_use_film", True))
    print(f"{rel_to_root(args.ckpt)}: latent_dim={cfg.get('latent_dim')} has_z={has_z}", flush=True)

    cat = collect(model, cfg, args.eps_per_cell)
    m_true = cat["mass"]
    cells = np.unique(np.round(m_true, 3))
    # per kg payload terms of each frame: exact because the cube geometry is fixed (see docstring)
    A = cat["H_pay"] / m_true[:, None, None]
    a = cat["g_pay"] / m_true[:, None]

    rng = np.random.default_rng(0)
    bank = rng.choice(len(m_true), size=min(args.bank, len(m_true)), replace=False)
    q_b = torch.from_numpy(cat["q_abs"][bank]).to(DEVICE).float()
    dq_b = torch.from_numpy(cat["dq"][bank]).to(DEVICE).float()
    H_arm_b, g_arm_b = cat["H_arm"][bank].mean(0), cat["g_arm"][bank].mean(0)
    A_b, a_b = A[bank].mean(0), a[bank].mean(0)

    sw = {k: [] for k in ("mass", "z_bar", "H_model", "g_model", "H_gt", "g_gt", "n")}
    re = {k: [] for k in ("mass", "n", "dH", "dg", "H_full", "g_full", "H_model", "g_model")}
    with torch.no_grad():
        for c in cells:
            k = np.round(m_true, 3) == c
            z_bar = cat["z"][k].mean(0)
            z = (torch.from_numpy(z_bar)[None, :].to(DEVICE).float().expand(len(bank), -1) if has_z else None)
            t = model.core(q_b, dq_b, z=z)
            sw["mass"].append(c); sw["n"].append(int(k.sum())); sw["z_bar"].append(z_bar)
            sw["H_model"].append(t.H.double().cpu().numpy().mean(0))
            sw["g_model"].append(t.g.double().cpu().numpy().mean(0))
            sw["H_gt"].append(H_arm_b + c * A_b)
            sw["g_gt"].append(g_arm_b + c * a_b)
            # real z, pose-matched: the frames of this cell, at their own mass and their own z
            re["mass"].append(c); re["n"].append(int(k.sum()))
            re["dH"].append((cat["H_model"][k] - cat["H_arm"][k] - cat["H_pay"][k]).mean(0))
            re["dg"].append((cat["g_model"][k] - cat["g_arm"][k] - cat["g_pay"][k]).mean(0))
            re["H_full"].append((cat["H_arm"][k] + cat["H_pay"][k]).mean(0))
            re["g_full"].append((cat["g_arm"][k] + cat["g_pay"][k]).mean(0))
            re["H_model"].append(cat["H_model"][k].mean(0))
            re["g_model"].append(cat["g_model"][k].mean(0))

    npz = {f"sweep_{k}": np.asarray(v) for k, v in sw.items()}
    npz.update({f"real_{k}": np.asarray(v) for k, v in re.items()})
    npz.update(bank_H_arm=H_arm_b, bank_g_arm=g_arm_b, bank_A=A_b, bank_a=a_b,
               bank_q=cat["q_abs"][bank], bank_dq=cat["dq"][bank], bank_mass=m_true[bank])
    np.savez_compressed(out_dir / "hg_vs_mass.npz", **npz)
    json.dump({"row": args.row, "checkpoint": rel_to_root(args.ckpt),
               "has_z": has_z, "latent_dim": cfg.get("latent_dim"),
               "film_modulation_limit": cfg.get("delan_film_modulation_limit"),
               "film_temperature": cfg.get("delan_film_temperature"),
               "n_frames": int(len(m_true)), "n_bank": int(len(bank)),
               "eps_per_cell": args.eps_per_cell, "cells_kg": [float(c) for c in cells]},
              open(out_dir / "hg_vs_mass_meta.json", "w"), indent=1)

    print(f"\nbank {len(bank)} poses | per kg payload A: H_22 {A_b[1, 1]:.3f} kg m2/kg, a: g_2 {a_b[1]:.2f} N m/kg")
    print(f"{'m':>5s} | {'H22 gt':>7s} {'model':>7s} | {'g2 gt':>7s} {'model':>7s}")
    for i, c in enumerate(cells):
        print(f"{c:5.2f} | {sw['H_gt'][i][1, 1]:7.3f} {sw['H_model'][i][1, 1]:7.3f} | "
              f"{sw['g_gt'][i][1]:7.2f} {sw['g_model'][i][1]:7.2f}")
    report()
    print(f"-> {out_dir}/hg_vs_mass.npz")


if __name__ == "__main__":
    main()
