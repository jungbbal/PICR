"""Per-frame Lagrangian terms of a checkpoint with the history-torque lag of its config.

Wraps cdelan.analysis.mcg_vs_gt.episode_terms (teacher-forced H, c, g of the model next to the simulator's arm terms and
the analytic payload terms, at every phase-3 frame).

The library function keeps frame t if rows t-12..t are phase 3 and encodes z_t from state[t-12:t] with the torques
tau[t-12:t] of the same rows. A checkpoint with history_torque_lag L (PICR: L = 1) reads the same state
window but the torques tau[t-12-L : t-L]. Here
  * the frame set is narrowed (for lag-0 checkpoints too) to frames whose rows t-12-ANCHOR_LAG..t are phase 3, so
    every checkpoint is scored on the same frames, pose banks and simulator references;
  * for a lagged checkpoint H, c, g and z are recomputed with the lagged torque window.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch

import cdelan.analysis.mcg_vs_gt as M

ANCHOR_LAG = 1
STATS = {"frames_orig": 0, "frames_kept": 0, "asserted": False}


def episode_terms(model, cfg, path, name, z_override=None):
    r = M.episode_terms(model, cfg, path, name, z_override=z_override)
    if r is None:
        return None
    hist = int(cfg["history_len"])
    lag = int(cfg.get("history_torque_lag", 0) or 0)
    assert lag <= ANCHOR_LAG
    shift = 1 if cfg.get("align_object_clock") else 0
    rr = int(cfg.get("robot_dof", 9))
    with h5py.File(path, "r") as fh:
        ep = fh["data"][name]
        q_rel = np.asarray(ep["obs/joint_pos"], dtype=np.float64)[shift:, :rr]
        dq = np.asarray(ep["obs/joint_vel"], dtype=np.float64)[shift:, :rr]
        tau = np.asarray(ep[f"robot_torques/{cfg['torque_key']}"], dtype=np.float64)[shift:, :rr]
        phase = np.asarray(ep["obs/phase"]).reshape(-1)[shift:]
        obj_pose = np.asarray(ep["states/rigid_object/object/root_pose"], dtype=np.float64)
    idx = r["idx"]
    lo = idx - hist - ANCHOR_LAG
    keep = lo >= 0
    keep[keep] &= phase[lo[keep]] == 3
    n0 = len(idx)
    STATS["frames_orig"] += n0
    STATS["frames_kept"] += int(keep.sum())
    if keep.sum() < 5:
        return None
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n0 else v) for k, v in r.items()}
    if lag and getattr(model, "use_context_encoder", True) and z_override is None:
        T = min(len(q_rel), len(obj_pose), len(phase))
        state = M._assemble_history_states(model, q_rel[:T], dq[:T], obj_pose[:T], ep_path=path, name=name, cfg=cfg, T=T)
        tau = tau[:T]
        b_all = out["idx"]
        Hm, cm, gm, zs = [], [], [], []
        with torch.no_grad():
            for s in range(0, len(b_all), 256):
                b = b_all[s:s + 256]
                hs = torch.stack([state[t - hist: t] for t in b]).to(M.DEVICE).float()
                ht = torch.stack([torch.from_numpy(tau[t - hist - lag: t - lag]) for t in b]).to(M.DEVICE).float()
                if lag == 1 and not STATS["asserted"]:
                    t0 = int(b[0])
                    assert np.array_equal(ht[0].double().cpu().numpy(), tau[t0 - hist - 1: t0 - 1])
                    assert np.array_equal(ht[0, 1:].double().cpu().numpy(), tau[t0 - hist: t0][:-1])
                    assert hs.shape[1] == hist
                    assert np.array_equal(hs[0].double().cpu().numpy(), state[t0 - hist: t0].numpy())
                    print(f"[lagterms] frame t={t0}: states rows {t0 - hist}..{t0 - 1}, "
                          f"torques rows {t0 - hist - 1}..{t0 - 2}", flush=True)
                    STATS["asserted"] = True
                z = model.encode_context(hs, ht).mu
                terms = model.core(torch.from_numpy(out["q_abs"][s:s + 256]).to(M.DEVICE).float(),
                                   torch.from_numpy(dq[b]).to(M.DEVICE).float(),
                                   z=z if model.delan_use_film else None)
                Hm.append(terms.H.double().cpu()); cm.append(terms.coriolis.double().cpu())
                gm.append(terms.g.double().cpu()); zs.append(z.double().cpu())
        out["H_model"] = torch.cat(Hm).numpy(); out["c_model"] = torch.cat(cm).numpy()
        out["g_model"] = torch.cat(gm).numpy(); out["z"] = torch.cat(zs).numpy()
    return out


def report():
    print(f"[lagterms] frames: library filter {STATS['frames_orig']}, kept {STATS['frames_kept']}, "
          f"dropped {STATS['frames_orig'] - STATS['frames_kept']}", flush=True)
