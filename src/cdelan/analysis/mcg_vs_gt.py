#!/usr/bin/env python3
"""Are the model's M, C, G the *real* M, C, G — or just a decomposition that sums right?

The model is never supervised on H, C or g individually. It is supervised on the state
transition, and any (H, C, g) triple that reproduces the transition on the data manifold is
an optimum of that objective. This script asks whether the one it found is the physical one.

Ground truth is not reconstructed from a URDF: Isaac logs it per timestep in
``robot_dynamics/{mass_matrix,coriolis,gravity}``. But it logs the ARM ALONE -- the payload
is a separate rigid body, not part of the articulation, so ``corr(m_obj, gravity_j) ~ 0``
on every joint. The payload's whole effect sits in ``tau_applied - inverse_dynamics_tau``.

So the comparison target is built in two pieces:

    GT_arm      logged, payload-free
    payload     analytic, from the KNOWN object mass/inertia and the rigid grasp:
                    dH = m Jc^T Jc + Jw^T I_w Jw,   Jc_j = Jv_j + Jw_j x lever
                    dg = m g0 d(h_com)/dq
                    dc = Christoffel(dH)
    GT_full  =  GT_arm + payload          <- what a correct model must equal

The payload piece is validated, not assumed: on phase-3 frames with |dq| < 0.24 rad/s the
analytic ``dg`` predicts the measured residual with slope 0.985/0.998/1.042 and r > 0.99 on
joints 1/3/5. Above that speed the residual also carries the payload's inertial reaction,
which is what dH and dc are for.

Three questions, three outputs:

  1  per-term time series (``--episode``)   model vs GT_arm vs GT_full, joint by joint
  2  per-term error table (``--n_episodes``) relative RMSE and R^2 against GT_full,
     against GT_arm, and -- the control -- the error of predicting GT_full with GT_arm,
     which is the payload signal the model had to find in the first place
  3  the z-increment test: f(q, z_ep) - f(q, z_mean) is the model's OWN estimate of what
     the payload adds. Compared against the analytic payload increment it gives a
     unit-carrying answer, in N*m and kg*m^2, to "does z mean mass".

Library module here: 2_mcg/torque_decomp_substep.py imports its term helpers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
from . import _plotstyle

_plotstyle.apply(0.9)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cdelan.data.dataset import load_all_episode_refs, resolve_split_refs
from cdelan.eval.rollout import load_checkpoint_model
from cdelan.models.utils import FrankaForwardKinematics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G0 = 9.81
N_ARM = 7
DT_FD = 1.0e-5  # central-difference step for dH/dq, in radians

C_MODEL, C_GT, C_ARM = "#2a78d6", "#0b0b0b", "#b0aca2"
GRIDLINE = "#dedcd6"


# --------------------------------------------------------------------------- #
# the payload's analytic contribution to H, c and g
# --------------------------------------------------------------------------- #
def _quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """(T, 4) wxyz -> (T, 3, 3). Isaac Lab's root_pose quaternion is scalar-FIRST; the PhysX
    tensor API's ``get_coms`` is scalar-last. Do not share one converter between them."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _skew_cross(Jw: torch.Tensor, lever: torch.Tensor) -> torch.Tensor:
    """Column-wise ``Jw_j x lever`` for a (T, 3, n) Jacobian block and a (T, 3) lever."""
    n = Jw.shape[-1]
    return torch.cross(
        Jw.transpose(1, 2), lever[:, None, :].expand(-1, n, -1), dim=-1
    ).transpose(1, 2)  # (T, 3, n)


def payload_inertia(fk, q_abs: torch.Tensor, mass: torch.Tensor,
                    lever_grip: torch.Tensor, inertia_grip: torch.Tensor) -> torch.Tensor:
    """H the payload adds, as a function of q. Everything held fixed is in the GRIPPER frame,
    so it rotates with the arm exactly as the rigid grasp says it must."""
    _, R = fk(q_abs)
    J = fk.jacobian(q_abs)
    Jv, Jw = J[:, :3], J[:, 3:]
    lever_w = torch.einsum("tij,tj->ti", R, lever_grip)
    Jc = Jv + _skew_cross(Jw, lever_w)
    I_w = torch.einsum("tij,tjk,tlk->til", R, inertia_grip, R)
    return mass[:, None, None] * torch.einsum("tai,taj->tij", Jc, Jc) + torch.einsum(
        "tai,tab,tbj->tij", Jw, I_w, Jw
    )


def payload_gravity(fk, q_abs: torch.Tensor, mass: torch.Tensor,
                    lever_grip: torch.Tensor) -> torch.Tensor:
    """dV/dq for V = m g0 * (height of the payload CoM). Validated against tau - tau_ID."""
    _, R = fk(q_abs)
    J = fk.jacobian(q_abs)
    lever_w = torch.einsum("tij,tj->ti", R, lever_grip)
    dh = J[:, 2, :] + _skew_cross(J[:, 3:], lever_w)[:, 2, :]
    return (mass[:, None] * G0) * dh


def payload_terms(fk, q_abs: torch.Tensor, dq: torch.Tensor, mass: torch.Tensor,
                  lever_grip: torch.Tensor, inertia_grip: torch.Tensor):
    """(dH, dc, dg). dH/dq is central-differenced: the function is analytic and smooth, and
    float64 finite differences at 1e-5 rad are exact to ~1e-9 -- cheaper and more robust
    than pushing functorch through the FK's in-place index writes."""
    H = payload_inertia(fk, q_abs, mass, lever_grip, inertia_grip)
    g = payload_gravity(fk, q_abs, mass, lever_grip)

    dof = q_abs.shape[1]
    dH_dq = torch.zeros(q_abs.shape[0], dof, dof, dof, dtype=q_abs.dtype)
    for k in range(N_ARM):  # finger columns of J are zero, so their derivative is too
        e = torch.zeros_like(q_abs)
        e[:, k] = DT_FD
        plus = payload_inertia(fk, q_abs + e, mass, lever_grip, inertia_grip)
        minus = payload_inertia(fk, q_abs - e, mass, lever_grip, inertia_grip)
        dH_dq[:, :, :, k] = (plus - minus) / (2.0 * DT_FD)

    # c_k = sum_ij (dH_kj/dq_i - 0.5 dH_ij/dq_k) dq_i dq_j -- the same Christoffel
    # contraction DeLaNCore.coriolis_centrifugal performs on its own L.
    term1 = torch.einsum("tkji,ti,tj->tk", dH_dq, dq, dq)
    term2 = torch.einsum("tijk,ti,tj->tk", dH_dq, dq, dq)
    return H, term1 - 0.5 * term2, g


# --------------------------------------------------------------------------- #
# one episode: model terms, GT_arm terms, payload terms, all on one clock
# --------------------------------------------------------------------------- #
def episode_terms(model, cfg, path: str, name: str, z_override: torch.Tensor | None = None):
    """All three decompositions at the GROUND-TRUTH (q, dq) of every phase-3 frame.

    Teacher-forced on purpose: a rollout would compare terms at states the model drifted
    into, which confounds "is H right" with "did the trajectory diverge".
    """
    hist = int(cfg["history_len"])
    with h5py.File(path, "r") as fh:
        ep = fh["data"][name]
        # robot_dynamics/* is logged with states/*; obs/* lags it by exactly one index
        # (states/joint_position[t] == obs/joint_pos[t+1] + const, std 0.0). The model
        # eats obs-clock joint_pos, so drop the leading obs sample to put both on the
        # states clock -- the same shift TrainConfig.align_object_clock applies.
        shift = 1 if cfg.get("align_object_clock") else 0
        # The model's own generalized-coordinate width. 7 drops the two gripper fingers from
        # the system entirely (the 7-dof arms); 9 keeps them, which is what every recording
        # holds. Reading it from the checkpoint keeps one code path for both.
        r = int(cfg.get("robot_dof", 9))
        q_rel = np.asarray(ep["obs/joint_pos"], dtype=np.float64)[shift:, :r]
        dq = np.asarray(ep["obs/joint_vel"], dtype=np.float64)[shift:, :r]
        tau = np.asarray(ep[f"robot_torques/{cfg['torque_key']}"], dtype=np.float64)[shift:, :r]
        phase = np.asarray(ep["obs/phase"]).reshape(-1)[shift:]
        H_gt = np.asarray(ep["robot_dynamics/mass_matrix"], dtype=np.float64)
        c_gt = np.asarray(ep["robot_dynamics/coriolis"], dtype=np.float64)
        g_gt = np.asarray(ep["robot_dynamics/gravity"], dtype=np.float64)
        qdd_gt = np.asarray(ep["robot_dynamics/qdd"], dtype=np.float64)
        mass = np.asarray(ep["object_dynamics/mass"], dtype=np.float64).reshape(-1)
        inertia = np.asarray(ep["object_dynamics/inertia"], dtype=np.float64).reshape(-1, 3, 3)
        obj_pose = np.asarray(ep["states/rigid_object/object/root_pose"], dtype=np.float64)
        # The CoM offset in the object's BODY frame. Isaac logs root_pose at the object's LINK
        # origin while the inertia is about its CoM, and the two differ by exactly this vector
        # once the CoM is moved off centre. Datasets that never move it (randmass) hide the
        # distinction; studies 07/12/14 move it by up to 19.6 mm per axis, which is the whole
        # experiment. Using the link origin as the gravity lever misstates the payload's
        # gravity by up to 0.58 N*m -- ~2% of it on j1/j3, but 30-180% on the small joints.
        obj_com_body = np.asarray(ep["object_dynamics/com"], dtype=np.float64)[:, :3]

    T = min(len(q_rel), len(H_gt), len(mass), len(obj_pose), len(obj_com_body))
    sl = slice(0, T)
    q_rel, dq, tau, phase = q_rel[sl], dq[sl], tau[sl], phase[sl]
    # The logged arm terms are always 9-wide; narrow them to the model's width so the
    # payload terms (built at width r below) add without broadcasting. At r = 9 this is a
    # no-op. Locking the fingers reduces the system to the PRINCIPAL SUBMATRIX M[:7,:7] --
    # not a Schur complement -- because they are position-controlled and stalled against the
    # payload, not free; the finger links' own mass and gravity are already inside it.
    H_gt, c_gt, g_gt, qdd_gt = H_gt[sl, :r, :r], c_gt[sl, :r], g_gt[sl, :r], qdd_gt[sl, :r]
    mass, inertia, obj_pose = mass[sl], inertia[sl], obj_pose[sl]
    obj_com_body = obj_com_body[sl]

    # A separate double-precision FK: model.fk.double() would mutate the shared model,
    # and assemble_state feeds it float tensors elsewhere.
    fk = FrankaForwardKinematics(
        robot_dof=r, tool_z_offset=float(cfg.get("tool_z_offset", 0.1034))
    ).double()
    default = model.default_joint_pos.detach().cpu().numpy().astype(np.float64)
    q_abs = torch.from_numpy(q_rel + default[None, :])

    # The grasp, in the gripper frame, so it is a constant the FK carries around.
    p_grip, R_grip = fk(q_abs)
    # Gripper -> payload CoM, not -> link origin. payload_gravity integrates m*g0*height OF THE
    # CoM and payload_inertia's m*Jc'Jc term is the CoM velocity Jacobian (tests/test_mcg_vs_gt
    # checks dH against the kinetic energy about exactly this point), so the lever has to end at
    # the CoM or both terms are wrong by the offset.
    R_obj = _quat_to_R(torch.from_numpy(obj_pose[:, 3:7]))
    com_w = torch.from_numpy(obj_pose[:, :3]) + torch.einsum(
        "tij,tj->ti", R_obj, torch.from_numpy(obj_com_body)
    )
    lever_w = com_w - p_grip
    lever_grip = torch.einsum("tji,tj->ti", R_grip, lever_w)
    # object_dynamics/inertia is about the CoM in the object's BODY frame -- it is constant
    # through an episode while the object turns 55-80 deg, which a world-frame tensor could
    # not be. So rotate BODY -> world with R_obj before world -> gripper with R_grip. Every
    # mass/CoM dataset has an isotropic I (= m a^2/6 * eye), where this is exactly a no-op;
    # it only bites on a dataset with a real inertia axis, e.g. lift_virtualbody_phase3
    # (diagonal ratio up to 5.8), where skipping R_obj silently mis-orients the tensor.
    inertia_w = torch.einsum("tij,tjk,tlk->til", R_obj, torch.from_numpy(inertia), R_obj)
    inertia_grip = torch.einsum("tji,tjk,tkl->til", R_grip, inertia_w, R_grip)

    dH, dc, dg = payload_terms(
        fk, q_abs, torch.from_numpy(dq), torch.from_numpy(mass), lever_grip, inertia_grip
    )

    # The same payload with its CoM put back at the lattice centre (obj_com_body = 0), i.e.
    # the lever ends at the object's LINK origin instead of at its CoM. Everything else --
    # mass, inertia tensor about the CoM, kinematics -- is held. The difference between this
    # and the block above is therefore the CoM axis alone, in N*m and kg m^2, which is what
    # the per-axis z-increment needs and what a mass rescaling cannot supply.
    lever_grip_ref = torch.einsum(
        "tji,tj->ti", R_grip, torch.from_numpy(obj_pose[:, :3]) - p_grip
    )
    dH_ref, dc_ref, dg_ref = payload_terms(
        fk, q_abs, torch.from_numpy(dq), torch.from_numpy(mass), lever_grip_ref, inertia_grip
    )

    # --- model terms, one z per frame from the preceding history window ---------- #
    idx = np.array([t for t in range(hist, T) if (phase[t - hist : t + 1] == 3).all()])
    if len(idx) < 5:
        return None

    state = _assemble_history_states(model, q_rel, dq, obj_pose, ep_path=path, name=name,
                                     cfg=cfg, T=T)
    Hm, cm, gm, zs = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s : s + 256]
            hs = torch.stack([state[t - hist : t] for t in b]).to(DEVICE).float()
            ht = torch.stack([torch.from_numpy(tau[t - hist : t]) for t in b]).to(DEVICE).float()
            if not getattr(model, "use_context_encoder", True):
                z = hs.new_zeros(len(b), 0)  # context-free model: no latent
            elif z_override is None:
                z = model.encode_context(hs, ht).mu
            else:
                z = z_override[None, :].expand(len(b), -1).to(DEVICE).float()
            terms = model.core(
                torch.from_numpy(q_abs.numpy()[b]).to(DEVICE).float(),
                torch.from_numpy(dq[b]).to(DEVICE).float(),
                z=z if model.delan_use_film else None,
            )
            Hm.append(terms.H.double().cpu()); cm.append(terms.coriolis.double().cpu())
            gm.append(terms.g.double().cpu()); zs.append(z.double().cpu())

    out = dict(
        idx=idx, mass=mass[idx], phase=phase[idx], tau=tau[idx], qdd=qdd_gt[idx],
        H_model=torch.cat(Hm).numpy(), c_model=torch.cat(cm).numpy(), g_model=torch.cat(gm).numpy(),
        z=torch.cat(zs).numpy(),
        H_arm=H_gt[idx], c_arm=c_gt[idx], g_arm=g_gt[idx],
        q_abs=q_abs.numpy()[idx], lever=lever_grip.numpy()[idx],
        H_pay=dH.numpy()[idx], c_pay=dc.numpy()[idx], g_pay=dg.numpy()[idx],
        H_pay_com0=dH_ref.numpy()[idx], g_pay_com0=dg_ref.numpy()[idx],
        com_body=obj_com_body[idx],
        dq=dq[idx],
    )
    out["H_full"] = out["H_arm"] + out["H_pay"]
    out["c_full"] = out["c_arm"] + out["c_pay"]
    out["g_full"] = out["g_arm"] + out["g_pay"]
    return out


def _assemble_history_states(model, q_rel, dq, obj_pose, *, ep_path, name, cfg, T):
    """The public state the context encoder was trained on: ``2 * robot_dof + 13``, so 31 at
    9 dof and 27 at 7. The width follows ``q_rel``/``dq``, which the caller already narrowed.
    With ``context_input_arm_only`` the encoder reads only the leading ``2 * robot_dof``
    columns, but the shape check in encode_context still wants the full width."""
    with h5py.File(ep_path, "r") as fh:
        ep = fh["data"][name]
        vel = np.asarray(ep["states/rigid_object/object/root_velocity"], dtype=np.float64)[:T]
        origin = np.asarray(ep["initial_state/articulation/robot/root_pose"],
                            dtype=np.float64)[0, :3]
    pos = obj_pose[:T, :3] - (origin[None, :] if cfg.get("subtract_env_origin", True) else 0.0)
    state = np.concatenate([q_rel[:T], dq[:T], pos, obj_pose[:T, 3:7], vel[:, :3], vel[:, 3:6]],
                           axis=-1)
    return torch.from_numpy(state)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(pred: np.ndarray, target: np.ndarray) -> dict:
    """Relative RMSE against the target's own scale, and R^2, over the arm joints."""
    p, t = np.asarray(pred, float), np.asarray(target, float)
    err = p - t
    rmse = float(np.sqrt((err ** 2).mean()))
    denom = float(np.sqrt((t ** 2).mean()))
    var = float(((t - t.mean()) ** 2).mean())
    return {
        "rmse": rmse,
        "rel_rmse": rmse / max(denom, 1e-12),
        "r2": float(1.0 - (err ** 2).mean() / max(var, 1e-12)),
    }


def per_joint_rms(x: np.ndarray) -> list:
    """RMS of each arm joint's entry (diagonal, for H). Scale is the first thing to look at:
    an R^2 of -88 says nothing about whether the term is 10x too big or 10x too small."""
    a = arm(np.asarray(x, float))
    if a.ndim == 3:
        a = np.diagonal(a, axis1=1, axis2=2)
    return [round(float(v), 4) for v in np.sqrt((a ** 2).mean(0))]


def arm(x: np.ndarray) -> np.ndarray:
    """Arm joints only. The two finger joints carry the grasp force, not the payload."""
    return x[..., :N_ARM, :N_ARM] if x.ndim == 3 else x[..., :N_ARM]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--episode", type=str, default=None, help="episode name for the time series")
    ap.add_argument("--joint", type=int, default=None,
                    help="joint drawn in the per-mass time series; default draws all seven "
                         "into mcg_timeseries_by_mass_j{0..6}.png")
    ap.add_argument("--mass_targets", type=float, nargs="+", default=None,
                    help="payload masses [kg] the per-mass columns aim at; "
                         "default spreads them over the split's own range")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = load_checkpoint_model(args.ckpt, device=DEVICE)
    model, cfg = loaded.model.eval(), dict(loaded.config.__dict__)
    from cdelan.training.train import resolve_hdf5_paths
    refs = resolve_split_refs(load_all_episode_refs(resolve_hdf5_paths(loaded.config)), cfg, "val")
    refs = refs[: args.n_episodes]
    print(f"{len(refs)} val episodes, latent_dim={cfg.get('latent_dim')}, "
          f"film_limit={cfg.get('delan_film_modulation_limit')}")

    rows, per_ep = [], []
    for path, name in refs:
        r = episode_terms(model, cfg, path, name)
        if r is None:
            continue
        per_ep.append((path, name, r))
        rows.append(r)
    if not rows:
        raise SystemExit("no phase-3 windows found")

    cat = {k: np.concatenate([r[k] for r in rows]) for k in
           ("H_model", "c_model", "g_model", "H_arm", "c_arm", "g_arm",
            "H_pay", "c_pay", "g_pay", "H_full", "c_full", "g_full", "mass", "z", "dq",
            "tau", "qdd", "q_abs", "lever")}

    metrics = {"n_episodes": len(rows), "n_frames": int(len(cat["mass"])),
               "checkpoint": args.ckpt,
               "film_modulation_limit": cfg.get("delan_film_modulation_limit"),
               "terms": {}}
    slow = np.linalg.norm(cat["dq"][:, :N_ARM], axis=1)
    slow_k = slow <= np.percentile(slow, 20)
    for term in ("H", "c", "g"):
        metrics["terms"][term] = {
            "rms_per_joint": {
                "model": per_joint_rms(cat[f"{term}_model"]),
                "gt_full": per_joint_rms(cat[f"{term}_full"]),
                "gt_arm": per_joint_rms(cat[f"{term}_arm"]),
                "payload": per_joint_rms(cat[f"{term}_pay"]),
            },
            "model_vs_full": score(arm(cat[f"{term}_model"]), arm(cat[f"{term}_full"])),
            "model_vs_arm": score(arm(cat[f"{term}_model"]), arm(cat[f"{term}_arm"])),
            # the control: how much of GT_full is the payload at all? If arm-only already
            # scores well the term carries no payload signal and the comparison is vacuous.
            "arm_vs_full": score(arm(cat[f"{term}_arm"]), arm(cat[f"{term}_full"])),
            # gravity is the only term identifiable at rest, so the slowest fifth of the
            # frames is where a wrong g cannot hide behind C.
            "model_vs_full_slowest_20pct": score(
                arm(cat[f"{term}_model"])[slow_k], arm(cat[f"{term}_full"])[slow_k]
            ),
        }

    # the sum, which is the only thing training ever constrained
    tau_model = np.einsum("tij,tj->ti", cat["H_model"], cat["qdd"]) + cat["c_model"] + cat["g_model"]
    tau_full = np.einsum("tij,tj->ti", cat["H_full"], cat["qdd"]) + cat["c_full"] + cat["g_full"]
    # NB the logged qdd is Isaac's instantaneous acceleration, self-consistent with its own
    # M and tau at the sim step but only r=0.36 against the decimated finite difference the
    # model integrates. It is the right quantity for checking GT_full against tau_applied,
    # and the wrong one for judging the model's H -- read model_vs_full with that in mind.
    metrics["inverse_dynamics_tau"] = {
        "model_vs_full": score(arm(tau_model), arm(tau_full)),
        "model_vs_applied": score(arm(tau_model), arm(cat["tau"])),
        "full_vs_applied": score(arm(tau_full), arm(cat["tau"])),
    }

    # --- per payload mass ---------------------------------------------------- #
    # Pooling hides the question the dataset was built to ask. A 0.1 kg payload barely
    # perturbs the arm; a 3.5 kg one doubles the load on joints 1 and 3. If the model
    # only works at one end of the range, the pooled number says so nowhere.
    edges = [0.1, 0.75, 1.5, 2.25, 3.0, 3.6]
    metrics["by_mass"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (cat["mass"] >= lo) & (cat["mass"] < hi)
        if k.sum() < 30:
            continue
        row = {"mass_kg": [lo, hi], "n_frames": int(k.sum()),
               "mass_mean": round(float(cat["mass"][k].mean()), 3)}
        for term in ("H", "c", "g"):
            row[term] = {
                "model_vs_full": score(arm(cat[f"{term}_model"])[k], arm(cat[f"{term}_full"])[k]),
                "arm_vs_full": score(arm(cat[f"{term}_arm"])[k], arm(cat[f"{term}_full"])[k]),
                "rms_model_j1": round(float(np.sqrt((arm(cat[f"{term}_model"])[k][:, 1, 1] ** 2).mean()
                                       if cat[f"{term}_model"].ndim == 3
                                       else (arm(cat[f"{term}_model"])[k][:, 1] ** 2).mean())), 3),
                "rms_full_j1": round(float(np.sqrt((arm(cat[f"{term}_full"])[k][:, 1, 1] ** 2).mean()
                                      if cat[f"{term}_full"].ndim == 3
                                      else (arm(cat[f"{term}_full"])[k][:, 1] ** 2).mean())), 3),
            }
        metrics["by_mass"].append(row)

    # --- z-increment: the model's own payload estimate, in physical units --------- #
    z_mean = torch.from_numpy(cat["z"].mean(0))
    has_z = cat["z"].shape[-1] > 0  # a context-free model has no z-increment to test
    inc_model_g, inc_true_g, inc_model_H, inc_true_H, masses = [], [], [], [], []
    for path, name, r in (per_ep[: min(len(per_ep), 20)] if has_z else []):
        base = episode_terms(model, cfg, path, name, z_override=z_mean)
        if base is None:
            continue
        inc_model_g.append(r["g_model"] - base["g_model"])
        inc_model_H.append(r["H_model"] - base["H_model"])
        # the analytic increment at the same q: payload(m_ep) - payload(m_mean)
        scale = (cat["mass"].mean() / r["mass"])[:, None]
        inc_true_g.append(r["g_pay"] * (1.0 - scale))
        inc_true_H.append(r["H_pay"] * (1.0 - scale)[:, :, None])
        masses.append(r["mass"])
    if inc_model_g:
        ig, it = np.concatenate(inc_model_g), np.concatenate(inc_true_g)
        iH, itH = np.concatenate(inc_model_H), np.concatenate(inc_true_H)
        a, b = arm(ig).ravel(), arm(it).ravel()
        slope = float(np.polyfit(b, a, 1)[0])
        metrics["z_increment"] = {
            "gravity": {**score(arm(ig), arm(it)), "slope_model_on_true": slope,
                        "corr": float(np.corrcoef(a, b)[0, 1])},
            "inertia": score(arm(iH), arm(itH)),
            "note": "slope is the fraction of the true payload effect z actually delivers",
        }

    # --- z-increment, split into the mass and CoM axes ---------------------------- #
    # The block above answers "how much of the payload's effect does z deliver" with a mass
    # rescaling, so it can only speak about mass. On a mass x CoM lattice the same question
    # has two answers, and they are not interchangeable: the CoM axis moves g by ~0.1 N*m
    # against mass's ~3 N*m, so a latent that carries mass perfectly and CoM not at all
    # scores ~100% on the block above.
    #
    # The two analytic directions, evaluated at each episode's own (q, dq):
    #   d_mass = g_pay(m_ep, c=0) - g_pay(m_bar, c=0)      mass at the lattice centre
    #   d_com  = g_pay(m_ep, c_ep) - g_pay(m_ep, c=0)      CoM at this episode's mass
    # They sum to the full increment, so this is a change of basis, not an approximation.
    # The model's own increment is the same z_mean counterfactual the block above uses, and
    # the two slopes come from ONE joint least squares -- fitting them separately would
    # credit each axis with the part the other explains.
    if inc_model_g:
        d_mass_g, d_com_g, d_model_g = [], [], []
        m_bar = float(cat["mass"].mean())
        for path, name, r in per_ep[: min(len(per_ep), 20)]:
            base = episode_terms(model, cfg, path, name, z_override=z_mean)
            if base is None:
                continue
            d_model_g.append(r["g_model"] - base["g_model"])
            d_mass_g.append(r["g_pay_com0"] * (1.0 - (m_bar / r["mass"])[:, None]))
            d_com_g.append(r["g_pay"] - r["g_pay_com0"])
        # Per-episode blocks, kept separate so the confidence interval can be resampled over
        # EPISODES. Frame-to-frame residuals inside one episode are heavily autocorrelated, so
        # an ordinary standard error over ~10^4 frames would be optimistic by an order of
        # magnitude. The CoM direction carries ~15x less torque than the mass one, so its
        # slope is the one that needs an honest interval.
        blocks = [(arm(m).ravel(), arm(c).ravel(), arm(g).ravel())
                  for m, c, g in zip(d_mass_g, d_com_g, d_model_g)]
        A = np.stack([np.concatenate([b[0] for b in blocks]),
                      np.concatenate([b[1] for b in blocks])], axis=1)
        y = np.concatenate([b[2] for b in blocks])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        rng = np.random.RandomState(0)
        draws = []
        for _ in range(400):
            pick = rng.randint(0, len(blocks), len(blocks))
            ab = np.stack([np.concatenate([blocks[i][0] for i in pick]),
                           np.concatenate([blocks[i][1] for i in pick])], axis=1)
            yb = np.concatenate([blocks[i][2] for i in pick])
            draws.append(np.linalg.lstsq(ab, yb, rcond=None)[0])
        draws = np.stack(draws)
        ci = np.percentile(draws, [2.5, 97.5], axis=0)
        pred = A @ beta
        ss_t = float(((y - y.mean()) ** 2).sum())
        metrics["z_increment_axes"] = {
            "reference": "mass at the lattice-centre CoM; CoM at this episode's own mass",
            "mean_mass_kg": m_bar,
            "delivers_mass": float(beta[0]),
            "delivers_com": float(beta[1]),
            "delivers_mass_ci95": [float(ci[0, 0]), float(ci[1, 0])],
            "delivers_com_ci95": [float(ci[0, 1]), float(ci[1, 1])],
            "n_episodes_bootstrapped": len(blocks),
            "r2_joint": float(1.0 - ((y - pred) ** 2).sum() / ss_t),
            "true_rms_nm": {"mass": float(np.sqrt((A[:, 0] ** 2).mean())),
                            "com": float(np.sqrt((A[:, 1] ** 2).mean()))},
            "corr_between_axes": float(np.corrcoef(A[:, 0], A[:, 1])[0, 1]),
            "com_body_std_mm": (np.concatenate([r["com_body"] for _, _, r in
                                                per_ep[: min(len(per_ep), 20)]]).std(0)
                                * 1e3).tolist(),
            "note": "slope on each analytic direction from one joint least squares; 1.0 "
                    "means z moves gravity exactly as much as that axis really does",
        }
        print(f"[z-increment] delivers  mass {beta[0]:6.3f} [{ci[0,0]:.3f}, {ci[1,0]:.3f}]   "
              f"CoM {beta[1]:6.3f} [{ci[0,1]:.3f}, {ci[1,1]:.3f}]   "
              f"(joint R2 {metrics['z_increment_axes']['r2_joint']:.3f}; true rms "
              f"{metrics['z_increment_axes']['true_rms_nm']['mass']:.3f} / "
              f"{metrics['z_increment_axes']['true_rms_nm']['com']:.3f} N*m, "
              f"{len(blocks)} episodes)")

    _plot_timeseries(per_ep, args.episode, os.path.join(args.out_dir, "mcg_timeseries.png"))
    # --joint unset draws all seven: one file per joint, since the mass columns already use
    # the width. --joint N keeps the old single figure under the old name.
    for j in (range(N_ARM) if args.joint is None else [args.joint]):
        suffix = "" if args.joint is not None else f"_j{j}"
        _plot_timeseries_by_mass(per_ep,
                                 os.path.join(args.out_dir, f"mcg_timeseries_by_mass{suffix}.png"),
                                 joint=j, targets=args.mass_targets)
    _plot_scatter(cat, os.path.join(args.out_dir, "mcg_scatter.png"))
    metrics["mass_sweep"] = _plot_by_mass(
        model, cat, os.path.join(args.out_dir, "mcg_by_mass.png")
    )

    with open(os.path.join(args.out_dir, "mcg_vs_gt.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics["mass_sweep"], indent=2))


def _plot_timeseries(per_ep, want: str | None, out: str, joints=None) -> None:
    """Every arm joint by default.

    This used to draw joints 1/3/5 -- the ones the payload's *weight* loads. That hides the
    wrist, which is where the model's M, C, G actually break (j6's H runs 90x the truth on
    applied_torque), so the plot could not show the failure it was there to diagnose.
    """
    pick = next((p for p in per_ep if want in (None, p[1])), per_ep[0])
    _, name, r = pick
    t = np.arange(len(r["idx"])) * 0.02
    joints = list(range(N_ARM)) if joints is None else list(joints)
    fig, axes = plt.subplots(3, len(joints), figsize=(2.45 * len(joints) + 0.6, 8.0),
                             sharex=True, squeeze=False)
    for col, j in enumerate(joints):
        for row, (term, label) in enumerate(
            (("g", "gravity\n$g_j$  [N·m]"), ("c", "Coriolis\n$(C\\dot q)_j$  [N·m]"),
             ("H", "inertia\n$H_{jj}$  [kg·m²]"))
        ):
            ax = axes[row][col]
            key = (lambda k: r[f"{term}_{k}"][:, j, j]) if term == "H" else \
                  (lambda k: r[f"{term}_{k}"][:, j])
            ax.plot(t, key("arm"), color=C_ARM, lw=1.6, label="GT arm only")
            ax.plot(t, key("full"), color=C_GT, lw=2.0, label="GT arm + payload")
            ax.plot(t, key("model"), color=C_MODEL, lw=1.6, ls="--", label="model")
            if row == 0:
                ax.set_title(f"joint {j}", fontsize=10)
            if col == 0:
                ax.set_ylabel(label)
            if row == 2 and col == len(joints) // 2:
                ax.set_xlabel("time in phase 3  [s]")
            ax.tick_params(labelsize=7.5)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{name}   payload {r['mass'][0]:.2f} kg", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print("wrote", out)


def _plot_timeseries_by_mass(per_ep, out: str, joint: int = 1,
                             targets=None, n_cols: int = 5) -> None:
    """The same three terms over time, one column per payload mass.

    Each column is one held-out episode carried through phase 3. Within a column nothing
    changes but the clock, so the two curves are directly comparable; across columns the
    only thing that changes is how heavy the object is. Black is arm + payload, the target a
    correct model must equal -- if the model tracked the payload, its dashed curve would sit
    on black in every column and stay there as the columns get heavier.

    The payload-free GT_arm curve is deliberately NOT drawn here: it is the reference for
    "how big is the payload signal", which the per-episode ``_plot_timeseries`` figure and
    the ``mcg_vs_gt.json`` error table already carry. This figure is only about whether the
    model reaches GT_full.
    """
    masses = np.array([r["mass"][0] for _, _, r in per_ep])
    if targets is None:
        # Spread the columns over the masses this dataset actually holds. A fixed ladder
        # tuned for the 0.1-3.5 kg sets picks the same heaviest episode several times on a
        # 0.1-2.0 kg one, which silently turns a mass sweep into repeated columns.
        targets = np.linspace(masses.min(), masses.max(), n_cols)
    picks, used = [], set()
    for want in targets:
        order = np.argsort(np.abs(masses - want))
        for i in order:
            if i not in used:
                used.add(int(i)); picks.append(int(i)); break

    rows = (("g", "gravity\n$g_{%d}$  [N·m]" % joint),
            ("c", "Coriolis\n$(C\\dot q)_{%d}$  [N·m]" % joint),
            ("H", "inertia\n$H_{%d%d}$  [kg·m²]" % (joint, joint)))
    # Text is sized for a slide, not for a page: this figure is read projected, where the
    # 8-10 pt labels it used to carry vanish. Panels grow only 20 % while the type more than
    # doubles, so the type is bigger RELATIVE to the panel -- that ratio is what carries.
    FS_SUP, FS_COL, FS_YLAB, FS_XLAB, FS_LEG, FS_TICK = 34, 28, 23, 28, 24, 20
    # y shared along each row: the columns are only comparable if the axis is. On the
    # inertia row that flattens the GT curve against the bottom, which is the finding.
    fig, axes = plt.subplots(3, len(picks), figsize=(3.4 * len(picks) + 2.0, 9.8),
                             sharex="col", sharey="row")
    for col, i in enumerate(picks):
        _, name, r = per_ep[i]
        t = np.arange(len(r["idx"])) * 0.02
        for row, (term, label) in enumerate(rows):
            ax = axes[row][col]
            take = (lambda k: r[f"{term}_{k}"][:, joint, joint]) if term == "H" else \
                   (lambda k: r[f"{term}_{k}"][:, joint])
            ax.plot(t, take("full"), color=C_GT, lw=3.4, label="GT arm + payload")
            ax.plot(t, take("model"), color=C_MODEL, lw=3.0, ls="--", label="model, from z")
            # Fewer, larger ticks: at this type size the default locator collides with itself.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 5, 10]))
            ax.tick_params(labelsize=FS_TICK, width=1.8, length=7)
            if row == 0:
                ax.set_title(f"{r['mass'][0]:.2f} kg", fontsize=FS_COL, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=FS_YLAB, fontweight="bold")
            if row == 2 and col == len(picks) // 2:
                ax.set_xlabel("time in phase 3  [s]", fontsize=FS_XLAB, fontweight="bold")
    fig.suptitle(f"joint {joint} — GT vs the M, C, G the context vector builds",
                 y=0.995, fontsize=FS_SUP, fontweight="bold")
    # tight_layout first, then the legend into the strip it reserved: an in-axes legend at
    # this type size is wider than the panel that would hold it.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=0.35, h_pad=0.8)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.945), frameon=False, fontsize=FS_LEG,
               handlelength=2.2, columnspacing=2.6)
    fig.savefig(out, dpi=170)
    print("wrote", out)


def _plot_by_mass(model, cat, out: str, n_bins: int = 7, bank: int = 500) -> None:
    """How much does each term actually move when the payload gets heavier?

    Pose is held fixed and only the payload changes, so nothing here is confounded by
    where in the workspace the arm happened to be. One bank of real recorded frames is
    evaluated at every mass:

        GT      payload(q, m) is exactly linear in m -- the object's inertia is rescaled
                with its mass in this dataset, so the whole increment scales together and
                the truth is a straight line through the origin by construction.
        model   term(q, z_b), with z_b the mean latent of the episodes in mass bin b.
                The model's own belief about what a heavier payload does.

    Two model curves, because magnitude alone can be earned in the wrong direction:
      * the projection onto the true increment -- the part the model actually delivers,
      * the full norm of its increment -- everything it moves, on-axis or not.
    A model that had the physics right would put both on the black line.
    """
    rng = np.random.default_rng(0)
    n = len(cat["mass"])
    pick = rng.choice(n, min(bank, n), replace=False)
    edges = np.quantile(cat["mass"], np.linspace(0.0, 1.0, n_bins + 1))
    centres, z_bins = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (cat["mass"] >= lo) & (cat["mass"] <= hi)
        if k.sum() < 20:
            continue
        centres.append(float(cat["mass"][k].mean()))
        z_bins.append(cat["z"][k].mean(0))
    centres = np.asarray(centres)

    q = torch.from_numpy(cat["q_abs"][pick]).to(DEVICE).float()
    dq = torch.from_numpy(cat["dq"][pick]).to(DEVICE).float()
    m_frame = cat["mass"][pick]

    model_terms, gt_terms = [], []
    with torch.no_grad():
        for m_b, z_b in zip(centres, z_bins):
            z = torch.from_numpy(z_b)[None, :].expand(len(pick), -1).to(DEVICE).float()
            t = model.core(q, dq, z=z if model.delan_use_film else None)
            model_terms.append({"g": arm(t.g.double().cpu().numpy()),
                                "c": arm(t.coriolis.double().cpu().numpy()),
                                "H": arm(t.H.double().cpu().numpy())})
            # payload(m) = (m / m_frame) * payload(m_frame): exact, since dH, dc and dg are
            # all linear in the inertial parameters and this dataset scales I with m.
            r = (m_b / m_frame)
            gt_terms.append({"g": arm(cat["g_arm"][pick] + cat["g_pay"][pick] * r[:, None]),
                             "c": arm(cat["c_arm"][pick] + cat["c_pay"][pick] * r[:, None]),
                             "H": arm(cat["H_arm"][pick] + cat["H_pay"][pick] * r[:, None, None])})

    curves: dict = {"mass_kg": [round(float(v), 3) for v in centres]}
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    titles = {"g": ("gravity  $g$", "N·m"), "c": ("Coriolis  $C\\dot q$", "N·m"),
              "H": ("inertia  $H$", "kg·m²")}
    for ax, term in zip(axes, ("g", "c", "H")):
        flat = lambda a: a.reshape(a.shape[0], -1)
        d_gt = np.stack([flat(t[term]) - flat(gt_terms[0][term]) for t in gt_terms])
        d_md = np.stack([flat(t[term]) - flat(model_terms[0][term]) for t in model_terms])
        norm_gt = np.linalg.norm(d_gt, axis=-1)                       # (bins, frames)
        unit = d_gt / np.clip(norm_gt[..., None], 1e-12, None)
        proj = (d_md * unit).sum(-1)
        ax.plot(centres, norm_gt.mean(1), color=C_GT, lw=2.0, marker="o", ms=4,
                label="true payload increment")
        ax.plot(centres, proj.mean(1), color=C_MODEL, lw=1.8, marker="s", ms=4,
                label="model, delivered on-axis")
        ax.plot(centres, np.linalg.norm(d_md, axis=-1).mean(1), color=C_MODEL, lw=1.2,
                ls="--", alpha=0.7, label="model, total motion")
        ax.axhline(0.0, color=GRIDLINE, lw=0.8, zorder=0)
        name, unit_s = titles[term]
        ax.set_title(f"{name}   [{unit_s}]")
        ax.set_xlabel("payload mass  [kg]")
        # the unit is already in the panel title; repeating it here overruns the margin
        ax.set_ylabel(f"increment vs {centres[0]:.2f} kg")
        curves[term] = {
            "true": [round(float(v), 4) for v in norm_gt.mean(1)],
            "model_on_axis": [round(float(v), 4) for v in proj.mean(1)],
            "model_total": [round(float(v), 4) for v in np.linalg.norm(d_md, axis=-1).mean(1)],
            "delivered_fraction": round(float(proj.mean(1)[-1] / max(norm_gt.mean(1)[-1], 1e-12)), 3),
            "on_axis_fraction": round(float(proj.mean(1)[-1]
                                      / max(np.linalg.norm(d_md, axis=-1).mean(1)[-1], 1e-12)), 3),
        }
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("same poses, only the payload changes", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print("wrote", out)
    return curves


def _plot_scatter(cat, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    for ax, term, unit in zip(axes, ("g", "c", "H"), ("N·m", "N·m", "kg·m²")):
        x = arm(cat[f"{term}_full"]).ravel()
        y = arm(cat[f"{term}_model"]).ravel()
        k = np.random.default_rng(0).choice(len(x), min(6000, len(x)), replace=False)
        ax.scatter(x[k], y[k], s=2, alpha=0.25, color=C_MODEL, edgecolors="none")
        lo, hi = np.percentile(x, [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=C_GT, lw=1.0)
        ax.set_xlabel(f"GT arm + payload  [{unit}]")
        ax.set_ylabel(f"model  [{unit}]")
        ax.set_title(f"{term}   $R^2$ = {score(y, x)['r2']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    main()
