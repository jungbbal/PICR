"""Panel (a): per 20 ms window, the simulator's 40-substep mean torque terms vs the model's M, C, G terms.

Not a next-state prediction metric. Every model term is evaluated teacher-forced at the ground-truth state of the window
start, and the inertial term multiplies the model's H by the TRUE window-mean acceleration. It asks only: does each
Lagrangian term the model carries put the same torque into the window as the simulator's own term did?

One window k = one control step of 20 ms = 40 PhysX substeps of 0.5 ms (i = 0..39, state s_i at the substep start).
Reference, from the dataset's `substep_dynamics/` group (only the OOD evaluation set official_ood_2to4kg stores it):

    T_M^sim = 1/40 sum_i  M_i qdd_i          qdd_i = (qd_{i+1} - qd_i) / 0.5 ms
    T_C^sim = 1/40 sum_i  c_i                c = C(q, qd) qd, PhysX coriolis/centrifugal vector
    T_G^sim = 1/40 sum_i  g_i

Isaac's M, c, g are the ARTICULATION's: the cube is a separate rigid body. The model is trained on the measured joint
torque, which carries the cube, so its terms contain the payload. The reference therefore adds the analytic rigid-grasp
payload terms of `cdelan.analysis.mcg_vs_gt` (dH, dc, dg), evaluated at every substep from that substep's q, qd, qdd and
cube pose and averaged the same way. 7-DOF: the arm block M_i[:7, :7] qdd_i[:7] (fingers are not in the model); the finger
coupling M_i[:7, 7:] qdd_i[7:] is reported in the meta json, not added.

Model (trapezoid integrator): one core evaluation per window at the START state, and ddq_k = H_k^-1 (tau_bar_k - c_k - g_k)
is the window-MEAN acceleration (dq_{k+1} - dq_k) / dt. So the model's window identity is H_k qbar_k + c_k + g_k = tau_bar_k:

    T_M^model = H(q_k, z_k) qbar_k,  qbar_k = (dq_{k+1} - dq_k) / dt  (= 1/40 sum_i qdd_i exactly)
    T_C^model = c(q_k, dq_k, z_k)
    T_G^model = g(q_k, z_k)

z_k from the 12 preceding windows (lagterms.episode_terms supplies H, c, g).

    python3 torque_decomp.py --ckpt checkpoints/picr.pt --out_dir work/fig_lagrangian/decomp_picr_ood
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OOD_SET, contract_path, h5_path, rel_to_root, val_names     # noqa: E402
from lagterms import episode_terms, report                                     # noqa: E402
from cdelan.analysis.mcg_vs_gt import DEVICE, N_ARM, _quat_to_R, payload_terms  # noqa: E402
from cdelan.eval.rollout import load_checkpoint_model                          # noqa: E402
from cdelan.models.utils import FrankaForwardKinematics                       # noqa: E402

TERMS = ("T_M", "T_C", "T_G", "T_sum")
N_SUB = 40
PAY_DEV = "cpu"  # mcg_vs_gt.payload_terms allocates its dH/dq on the CPU


def substep_reference(fk, path: str, name: str, shift: int, idx: np.ndarray):
    """Window means (len(idx), 7) of the arm + payload terms, on the model's (shifted) clock.

    Window t on the shifted clock starts at obs row t + shift and its 40 substeps are recorder row t + shift
    (the recorder writes step k's substeps on row k; sub/joint_pos[k, 0] == obs row k)."""
    rows = idx + shift
    with h5py.File(path, "r") as fh:
        ep = fh["data"][name]
        sd = ep["substep_dynamics"]
        mean = {k: np.asarray(sd["mean"][k], dtype=np.float64)[rows] for k in
                ("inertial_arm", "inertial_finger", "coriolis", "gravity", "tau", "valid")}
        q = np.asarray(sd["sub/joint_pos"], dtype=np.float64)[rows][..., :N_ARM]
        qd = np.asarray(sd["sub/joint_vel"], dtype=np.float64)[rows][..., :N_ARM]
        qdd = np.asarray(sd["sub/qdd"], dtype=np.float64)[rows][..., :N_ARM]
        pose = np.asarray(sd["sub/object_pose"], dtype=np.float64)[rows]
        mass = float(np.asarray(ep["object_dynamics/mass"]).reshape(-1)[0])
        inertia = np.asarray(ep["object_dynamics/inertia"], dtype=np.float64)[0].reshape(3, 3)
        com_body = np.asarray(ep["object_dynamics/com"], dtype=np.float64)[0, :3]
        tau_mean = np.asarray(ep["robot_torques/measured_joint_force_mean"], dtype=np.float64)[rows]
    assert np.allclose(mean["tau"], tau_mean), "substep tau mean != recorder measured_joint_force_mean"
    n = len(rows)
    flat = lambda a: torch.from_numpy(a.reshape(n * N_SUB, *a.shape[2:])).to(PAY_DEV)  # noqa: E731
    qa, dqa, qdda, pz = flat(q), flat(qd), flat(qdd), flat(pose)
    # same lever / inertia construction as mcg_vs_gt.episode_terms, per substep
    p_grip, R_grip = fk(qa)
    R_obj = _quat_to_R(pz[:, 3:7])
    com_w = pz[:, :3] + R_obj @ torch.as_tensor(com_body, device=PAY_DEV)
    lever_grip = torch.einsum("tji,tj->ti", R_grip, com_w - p_grip)
    I = torch.as_tensor(inertia, device=PAY_DEV)
    inertia_grip = torch.einsum("tji,tjk,tkl,tml,tmn->tin", R_grip, R_obj, I.expand(len(qa), 3, 3), R_obj, R_grip)
    m = torch.full((len(qa),), mass, dtype=torch.float64, device=PAY_DEV)
    dH, dc, dg = [], [], []
    for s in range(0, len(qa), 8192):
        sl = slice(s, s + 8192)
        a, b, c = payload_terms(fk, qa[sl], dqa[sl], m[sl], lever_grip[sl], inertia_grip[sl])
        dH.append(torch.einsum("tij,tj->ti", a, qdda[sl])); dc.append(b); dg.append(c)
    unflat = lambda x: torch.cat(x).reshape(n, N_SUB, N_ARM).mean(1).cpu().numpy()  # noqa: E731
    pay_M, pay_C, pay_G = unflat(dH), unflat(dc), unflat(dg)
    return {
        "T_M": mean["inertial_arm"][:, :N_ARM] + pay_M,
        "T_C": mean["coriolis"][:, :N_ARM] + pay_C,
        "T_G": mean["gravity"][:, :N_ARM] + pay_G,
        "pay_M": pay_M, "pay_C": pay_C, "pay_G": pay_G,
        "finger_M": mean["inertial_finger"][:, :N_ARM],
        "tau_mean": tau_mean[:, :N_ARM],
        "qbar_sub": qdd.mean(1),
        "valid": mean["valid"] > 0.5,
        "mass": np.full(n, mass),
    }


def model_terms(model, cfg, path, name):
    """H, c, g at every window start and the window-mean acceleration."""
    r = episode_terms(model, cfg, path, name)
    if r is None:
        return None
    shift = 1 if cfg.get("align_object_clock") else 0
    with h5py.File(path, "r") as fh:
        ep = fh["data"][name]
        dq = np.asarray(ep["obs/joint_vel"], dtype=np.float64)[shift:, :N_ARM]
        phase = np.asarray(ep["obs/phase"]).reshape(-1)[shift:]
    # the window t -> t+1 must exist and end in phase 3, as the trainer's windows do
    keep = (r["idx"] + 1 < len(dq))
    keep[keep] &= phase[r["idx"][keep] + 1] == 3
    idx = r["idx"][keep]
    qbar = (dq[idx + 1] - dq[idx]) / float(cfg["dt"])
    H = r["H_model"][keep]
    return idx, shift, {
        "T_M": np.einsum("tij,tj->ti", H, qbar), "T_C": r["c_model"][keep], "T_G": r["g_model"][keep],
        "qbar": qbar, "q_abs": r["q_abs"][keep][:, :N_ARM],
    }


def rmse_table(ref, mod, rms_floor):
    rows = []
    for term in TERMS:
        e = mod[term] - ref[term]
        for j in list(range(N_ARM)) + ["all"]:
            if j == "all":
                err, r = e.reshape(-1), ref[term].reshape(-1)
            else:
                err, r = e[:, j], ref[term][:, j]
            rmse, rms = float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean(r ** 2)))
            ok = rms >= rms_floor
            rows.append({"term": term, "joint": j if j == "all" else j + 1, "n_windows": len(err),
                         "rmse_Nm": rmse, "ref_rms_Nm": rms, "model_rms_Nm": float(np.sqrt(np.mean(
                             (mod[term].reshape(-1) if j == "all" else mod[term][:, j]) ** 2))),
                         "nrmse": rmse / rms if ok else float("nan"),
                         "nrmse_note": "" if ok else f"ref RMS < {rms_floor:g} N m: not normalised"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_file", default=None, help=f"default: $PICR_DATA_ROOT/raw/{OOD_SET}/{OOD_SET}.hdf5")
    ap.add_argument("--contract", default=None, help=f"split json, episodes from splits.val (default: {OOD_SET}.json)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rms_floor", type=float, default=1e-3,
                    help="N m; below this reference RMS the normalised RMSE is reported as NaN")
    args = ap.parse_args()
    h5 = args.dataset_file or str(h5_path(OOD_SET))
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = load_checkpoint_model(args.ckpt, device=DEVICE)
    model, cfg = loaded.model.eval(), dict(loaded.config.__dict__)
    assert cfg["ode_solver"] == "trapezoid" and int(cfg["robot_dof"]) == N_ARM, \
        "window semantics above are derived for the 7-DOF trapezoid (start-point) integrator only"
    assert getattr(model, "qdd_residual", None) is None, "model adds a qdd residual outside M, C, G"
    if args.contract:
        names = json.load(open(args.contract))["splits"]["val"]
        names = [n if isinstance(n, str) else n[-1] for n in names]
    else:
        names = val_names(OOD_SET)
    fk = FrankaForwardKinematics(robot_dof=N_ARM, tool_z_offset=float(cfg.get("tool_z_offset", 0.1034))).double()

    per_ep, refs, mods = [], [], []
    for i, name in enumerate(names):
        out = model_terms(model, cfg, h5, name)
        if out is None:
            continue
        idx, shift, mod = out
        ref = substep_reference(fk, h5, name, shift, idx)
        ok = ref["valid"]
        ref = {k: v[ok] for k, v in ref.items()}
        mod = {k: v[ok] for k, v in mod.items()}
        idx = idx[ok]
        # the substep mean of qdd_i telescopes to the window finite difference the model uses
        assert np.allclose(ref["qbar_sub"], mod["qbar"], atol=1e-3), name
        for d in (ref, mod):
            d["T_sum"] = d["T_M"] + d["T_C"] + d["T_G"]
        per_ep.append((name, idx, shift, ref, mod))
        refs.append(ref); mods.append(mod)
        if i % 20 == 0:
            print(f"{i}/{len(names)} {name}: {len(idx)} windows", flush=True)
    ref = {k: np.concatenate([r[k] for r in refs]) for k in refs[0]}
    mod = {k: np.concatenate([m[k] for m in mods]) for k in mods[0]}
    n_win = len(ref["T_M"])
    print(f"{len(per_ep)} episodes, {n_win} phase-3 windows")

    rows = [dict(model_eval="window_start", **r) for r in rmse_table(ref, mod, args.rms_floor)]
    with open(os.path.join(args.out_dir, "torque_decomp_rmse.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # reference-side diagnostics: how far the reference itself is from the measured torque
    res = ref["T_sum"] - ref["tau_mean"]
    diag = {
        "checkpoint": rel_to_root(args.ckpt), "dataset_file": Path(h5).name,
        "n_episodes": len(per_ep), "n_windows": n_win,
        "dt": float(cfg["dt"]), "n_substeps": N_SUB, "rms_floor_Nm": args.rms_floor,
        "ref_closure_rms_Nm_vs_measured_tau_mean": np.sqrt(np.mean(res ** 2, 0)).round(4).tolist(),
        "ref_closure_median_abs_Nm": np.median(np.abs(res), 0).round(4).tolist(),
        "finger_coupling_rms_Nm_not_in_T_M": np.sqrt(np.mean(ref["finger_M"] ** 2, 0)).round(4).tolist(),
        "payload_share_rms_Nm": {k: np.sqrt(np.mean(ref[f"pay_{k[-1]}"] ** 2, 0)).round(4).tolist()
                                 for k in ("T_M", "T_C", "T_G")},
        "model_sum_vs_measured_tau_rms_Nm": np.sqrt(np.mean((mod["T_sum"] - ref["tau_mean"]) ** 2, 0)).round(4).tolist(),
    }
    np.savez(os.path.join(args.out_dir, "torque_decomp_windows.npz"),
             episode=np.concatenate([[p[0]] * len(p[1]) for p in per_ep]),
             window_row=np.concatenate([p[1] + p[2] for p in per_ep]), mass=ref["mass"],
             **{f"ref_{k}": ref[k] for k in TERMS}, **{f"model_{k}": mod[k] for k in TERMS}, tau_mean=ref["tau_mean"])
    with open(os.path.join(args.out_dir, "torque_decomp_meta.json"), "w") as f:
        json.dump(diag, f, indent=1)
    for r in rows:
        if r["joint"] == "all":
            print(f"{r['term']:6s} all joints RMSE {r['rmse_Nm']:.3f} N m  nRMSE {r['nrmse']:.3f}")
    report()


if __name__ == "__main__":
    main()
