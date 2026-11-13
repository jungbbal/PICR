"""Panel (b): mid-carry payload switch 0.1 -> 2 kg -- do the Lagrangian terms computed from z_t move to the new payload's?

Fixed-state design (so a change in a term can only come from z, never from the arm moving):
    bank     B states (q, dq) sampled from the post-switch phase-3 frames of the switch episodes
    GT(m)    H_arm(q) + m A(q),  c_arm(q,dq) + m dc(q,dq),  g_arm(q) + m a(q)   -- exact: one cube geometry
    model    H(q, z_t), c(q, dq, z_t), g(q, z_t) on the SAME bank, z_t = the window-encoded context of the switch
             episode at frame t (12-frame history on that episode's own states)
    per (episode, t): for each term, mean over the bank of
        d_new = || model - GT(m_new) || / step,  d_old = || model - GT(m_old) || / step,  step = || GT(m_old) - GT(m_new) ||
The figure plots the representative entries (rep_*: H_22, g_2, ... on the bank state whose c step is largest) against the
frame relative to the switch, with their two GT levels (rep_gt_old / rep_gt_new). Also kept: the same distances at the REAL
state of every frame (real_*) and z_t itself.

Dataset: ppo_lat8orig_meantau_switch012_twin_lead_nodrop (every episode in splits.val of its contract).

    python3 switch_mcg.py --ckpt checkpoints/picr.pt --out work/fig_lagrangian/switch012_picr.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SWITCH_M_NEW, SWITCH_M_OLD, SWITCH_SET, h5_path, rel_to_root, val_names   # noqa: E402
from lagterms import episode_terms, report                                                   # noqa: E402
from cdelan.analysis.mcg_vs_gt import DEVICE, N_ARM                                          # noqa: E402
from cdelan.eval.rollout import load_checkpoint_model                                        # noqa: E402

M_OLD, M_NEW = SWITCH_M_OLD, SWITCH_M_NEW


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_file", default=None, help=f"default: $PICR_DATA_ROOT/raw/{SWITCH_SET}/{SWITCH_SET}.hdf5")
    ap.add_argument("--bank", type=int, default=64)
    ap.add_argument("--rel", type=int, nargs=2, default=[-70, 110], help="frames kept relative to the switch")
    a = ap.parse_args()
    H5 = a.dataset_file or str(h5_path(SWITCH_SET))
    mid = 0.5 * (M_OLD + M_NEW)
    out = Path(a.out)
    loaded = load_checkpoint_model(a.ckpt, device=DEVICE)
    model, cfg = loaded.model.eval(), dict(loaded.config.__dict__)
    assert int(cfg["robot_dof"]) == N_ARM
    has_z = bool(getattr(model, "use_context_encoder", True)) and bool(getattr(model, "delan_use_film", True))
    names = val_names(SWITCH_SET)
    print(f"{rel_to_root(a.ckpt)} has_z={has_z} latent={cfg.get('latent_dim')} | {len(names)} episodes", flush=True)

    # ---- pass 1: per-frame terms at the real states, z_t, switch row -------------------------------------
    rows = []                                                                   # one dict per episode
    t0 = time.time()
    with h5py.File(H5, "r") as f:
        mtr = {n: np.asarray(f["data"][n]["object_dynamics/mass"]).reshape(-1) for n in names}
        sgn = np.sign(M_NEW - M_OLD)
        t_switch = {n: int(np.argmax(sgn * (m - mid) > 0)) for n, m in mtr.items()}
        switched = {n: bool((sgn * (m - mid) > 0).any()) for n, m in mtr.items()}
    for i, n in enumerate(names):
        if not switched[n] or t_switch[n] < 20:
            continue
        r = episode_terms(model, cfg, H5, n)
        if r is None:
            continue
        rel = r["idx"] - t_switch[n]
        keep = (rel >= a.rel[0]) & (rel <= a.rel[1])
        if keep.sum() < 10:
            continue
        per_kg = {k: r[k + "_pay"] / r["mass"][:, None, None] if k == "H" else r[k + "_pay"] / r["mass"][:, None]
                  for k in ("H", "c", "g")}
        d = {"name": n, "rel": rel[keep], "mass": r["mass"][keep], "z": r["z"][keep],
             "q": r["q_abs"][keep], "dq": r["dq"][keep]}
        for k in ("H", "c", "g"):
            arm, pk, mod = r[k + "_arm"][keep], per_kg[k][keep], r[k + "_model"][keep]
            ax = (1, 2) if k == "H" else 1
            d[f"real_{k}_dnew"] = np.linalg.norm(mod - (arm + M_NEW * pk), axis=ax)
            d[f"real_{k}_dold"] = np.linalg.norm(mod - (arm + M_OLD * pk), axis=ax)
            d[f"real_{k}_dtrue"] = np.linalg.norm(mod - (arm + r["mass"][keep][..., None, None] * pk if k == "H"
                                                        else arm + r["mass"][keep][:, None] * pk), axis=ax)
            d[f"real_{k}_step"] = np.linalg.norm((M_OLD - M_NEW) * pk, axis=ax)
            d[f"real_{k}_arm"], d[f"real_{k}_perkg"], d[f"real_{k}_model"] = arm, pk, mod
        rows.append(d)
        if i % 40 == 0:
            print(f"  {i}/{len(names)} episodes, {time.time() - t0:.0f} s", flush=True)
    print(f"pass 1: {len(rows)} switched episodes kept, {time.time() - t0:.0f} s", flush=True)

    # ---- bank of fixed states: post-switch frames (rel >= 20) --------------------------------------------
    rng = np.random.default_rng(0)
    cand = [(ei, fi) for ei, d in enumerate(rows) for fi in np.where(d["rel"] >= 20)[0]]
    pick = [cand[j] for j in rng.choice(len(cand), size=min(a.bank, len(cand)), replace=False)]
    bank = {k: np.stack([rows[ei][src][fi] for ei, fi in pick])
            for k, src in (("q", "q"), ("dq", "dq"), ("H_arm", "real_H_arm"), ("c_arm", "real_c_arm"),
                           ("g_arm", "real_g_arm"), ("A", "real_H_perkg"), ("dc", "real_c_perkg"),
                           ("a", "real_g_perkg"))}
    B = len(pick)
    gt = {m: {"H": bank["H_arm"] + m * bank["A"], "c": bank["c_arm"] + m * bank["dc"],
              "g": bank["g_arm"] + m * bank["a"]} for m in (M_OLD, M_NEW)}
    step = {"H": np.linalg.norm(gt[M_OLD]["H"] - gt[M_NEW]["H"], axis=(1, 2)),
            "c": np.linalg.norm(gt[M_OLD]["c"] - gt[M_NEW]["c"], axis=1),
            "g": np.linalg.norm(gt[M_OLD]["g"] - gt[M_NEW]["g"], axis=1)}
    ri = int(np.argmax(step["c"]))                                           # representative fixed state
    print(f"bank {B} states | step {M_OLD}->{M_NEW} kg: H {step['H'].mean():.3f} kg m2, c {step['c'].mean():.3f}, "
          f"g {step['g'].mean():.3f} N m", flush=True)

    # ---- pass 2: every (episode, frame) z on the whole bank ---------------------------------------------
    q_b = torch.from_numpy(bank["q"]).to(DEVICE).float(); dq_b = torch.from_numpy(bank["dq"]).to(DEVICE).float()
    Z = np.concatenate([d["z"] for d in rows]) if has_z else np.zeros((sum(len(d["rel"]) for d in rows), 0))
    N = len(Z)
    fixed = {f"{k}_{w}": np.zeros(N) for k in ("H", "c", "g") for w in ("dnew", "dold")}
    rep = {k: np.zeros(N) for k in ("H22", "H44", "H66", "g2", "g4", "g6", "c2", "c4")}
    with torch.no_grad():
        for s in range(0, N, 64):
            z = torch.from_numpy(Z[s:s + 64]).to(DEVICE).float()                 # (n, L)
            n = len(z)
            qq = q_b[None].expand(n, -1, -1).reshape(-1, N_ARM); dd = dq_b[None].expand(n, -1, -1).reshape(-1, N_ARM)
            zz = z[:, None, :].expand(-1, B, -1).reshape(-1, z.shape[1]) if has_z else None
            t = model.core(qq, dd, z=zz if model.delan_use_film else None)
            H = t.H.double().cpu().numpy().reshape(n, B, N_ARM, N_ARM)
            c = t.coriolis.double().cpu().numpy().reshape(n, B, N_ARM)
            g = t.g.double().cpu().numpy().reshape(n, B, N_ARM)
            for k, v in (("H", H), ("c", c), ("g", g)):
                ax = (2, 3) if k == "H" else 2
                fixed[f"{k}_dnew"][s:s + n] = (np.linalg.norm(v - gt[M_NEW][k][None], axis=ax) / step[k][None]).mean(1)
                fixed[f"{k}_dold"][s:s + n] = (np.linalg.norm(v - gt[M_OLD][k][None], axis=ax) / step[k][None]).mean(1)
            rep["H22"][s:s + n] = H[:, ri, 1, 1]; rep["H44"][s:s + n] = H[:, ri, 3, 3]; rep["H66"][s:s + n] = H[:, ri, 5, 5]
            rep["g2"][s:s + n] = g[:, ri, 1]; rep["g4"][s:s + n] = g[:, ri, 3]; rep["g6"][s:s + n] = g[:, ri, 5]
            rep["c2"][s:s + n] = c[:, ri, 1]; rep["c4"][s:s + n] = c[:, ri, 3]
    print(f"pass 2 done, {time.time() - t0:.0f} s", flush=True)

    ep_id = np.concatenate([np.full(len(d["rel"]), i) for i, d in enumerate(rows)])
    npz = dict(episode=ep_id, name=np.array([d["name"] for d in rows]), rel=np.concatenate([d["rel"] for d in rows]),
               mass=np.concatenate([d["mass"] for d in rows]), z=Z, m_old=M_OLD, m_new=M_NEW,
               **{f"fixed_{k}": v for k, v in fixed.items()}, **{f"rep_{k}": v for k, v in rep.items()},
               rep_state=ri,
               rep_gt_old=np.array([gt[M_OLD]["H"][ri, 1, 1], gt[M_OLD]["H"][ri, 3, 3], gt[M_OLD]["H"][ri, 5, 5],
                                    gt[M_OLD]["g"][ri, 1], gt[M_OLD]["g"][ri, 3], gt[M_OLD]["g"][ri, 5],
                                    gt[M_OLD]["c"][ri, 1], gt[M_OLD]["c"][ri, 3]]),
               rep_gt_new=np.array([gt[M_NEW]["H"][ri, 1, 1], gt[M_NEW]["H"][ri, 3, 3], gt[M_NEW]["H"][ri, 5, 5],
                                    gt[M_NEW]["g"][ri, 1], gt[M_NEW]["g"][ri, 3], gt[M_NEW]["g"][ri, 5],
                                    gt[M_NEW]["c"][ri, 1], gt[M_NEW]["c"][ri, 3]]),
               rep_keys=np.array(list(rep)), bank_q=bank["q"], bank_dq=bank["dq"], bank_step_H=step["H"],
               bank_step_c=step["c"], bank_step_g=step["g"])
    for k in ("H", "c", "g"):
        for w in ("dnew", "dold", "dtrue", "step"):
            npz[f"real_{k}_{w}"] = np.concatenate([d[f"real_{k}_{w}"] for d in rows])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **npz)
    rel = npz["rel"]
    for k in ("H", "c", "g"):
        pre = fixed[f"{k}_dnew"][(rel < -5)].mean(); post = fixed[f"{k}_dnew"][(rel > 40)].mean()
        print(f"  fixed-state {k}: d_new/step pre {pre:.2f} -> post {post:.2f} | d_old/step pre "
              f"{fixed[f'{k}_dold'][rel < -5].mean():.2f} -> post {fixed[f'{k}_dold'][rel > 40].mean():.2f}")
    report()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
