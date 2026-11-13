#!/usr/bin/env python3
"""Dump PICR's predictions on the do(tau) fork pairs: every anchor window with its history, the simulator's
future and the model's 15-step open-loop prediction.

Each dataset episode is one branch of a fork: all branches of an env share the episode up to the anchor step k,
then branch b > 0 adds a constant torque step on one joint (branch 0 = null, no push). Episodes are cut to rows
k-13..k+15, because PICR's encoder torques lag the states by one frame (history_torque_lag 1): it sees the 12-frame
state history x[k-12..k-1], the torques tau[k-13..k-2] and the future torques from tau[k-1].

    python3 dump_traj.py --row picr --ckpt checkpoints/picr.pt --dataset official_intervene_jlm7000_h13 --out traj.npz

Output arrays, one row per (branch, env) anchor window:
    branch, env, mass, slip_m, fell, wall      bookkeeping + the clean-gate inputs (cube slip in the hand over
                                               the horizon, cube drop, velocity limit hit)
    hist_states (N, 12, D)  hist_torques (N, 12, 7)    the 12-frame history the model sees
    fut_states  (N, H, D)   fut_torques  (N, H, 7)     simulator future and the torques fed to the model
    pred_states (N, H, D)                              the model's open-loop prediction (its context encoder
                                                       reads this branch's own window)
    pred_frozen (N, H, D)                              same, but the context is encoded once from the NULL
                                                       branch's window and reused for every branch of that env
    q_abs_anchor (N, 9)                                recorded absolute joint position at the last history
                                                       frame, for forward kinematics
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from cdelan.analysis.predict import predict_batch
from cdelan.analysis.contract_eval import build_loader
from cdelan.eval.rollout import load_checkpoint_model

ROWS = ("picr",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="picr", choices=ROWS)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="dataset id, e.g. official_intervene_jlm7000_h13")
    ap.add_argument("--data_root", default=os.environ.get("PICR_DATA_ROOT", "data"),
                    help="datasets live at <data_root>/raw/<id>/<id>.hdf5")
    ap.add_argument("--contracts", default=os.environ.get("PICR_CONTRACTS", "data/contracts"))
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ID = a.dataset
    dataset_file = str(Path(a.data_root) / "raw" / ID / f"{ID}.hdf5")
    protocol = str(Path(a.contracts) / f"{ID}.json")
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    loaded = load_checkpoint_model(a.ckpt, device=dev)
    cfg = loaded.config
    cfg.external_split_json, cfg.external_split_val_key = protocol, "val"
    cfg.rollout_horizon = int(a.horizon)
    loader, n_ep, _ = build_loader(cfg, "val", a.batch_size, dataset_file)
    ds = loader.dataset
    dof, hist, H = cfg.robot_dof, int(cfg.history_len), cfg.rollout_horizon

    # the anchor window of every episode (its history ends one row before the anchor) + the clean-gate inputs
    want, info, flags, qabs = {}, {}, {}, {}
    names_by_path: dict[str, set] = {}
    for m in ds.sample_meta:
        names_by_path.setdefault(m[0], set()).add(m[1])
    for path, names in names_by_path.items():
        with h5py.File(path, "r") as fh:
            for name in names:
                ep = fh["data"][name]
                g = ep["intervention"]
                k = int(np.asarray(g["anchor_step"])[0, 0])
                b = int(np.asarray(g["branch"])[0, 0])
                e = int(np.asarray(g["env_index"])[0, 0])
                r = int(np.flatnonzero(np.asarray(g["step"]).reshape(-1).astype(int) == k)[0])
                z = np.asarray(ep["object_dynamics/root_pos_w"])[:, 2]
                cih = np.asarray(ep["grasp_offset/cube_in_hand"])
                sat = np.asarray(ep["obs/velocity_saturated"]).reshape(-1)
                hi = min(r + H, len(z) - 1)
                flags[(b, e)] = (float(np.abs(cih[r:hi + 1] - cih[r]).max()),
                                 bool((z[r:hi + 1] < z[r] - 0.03).any()), bool(sat[r:hi + 1].any()))
                want[(path, name)] = r - 1
                info[(path, name)] = (b, e)
                qabs[(b, e)] = np.asarray(ep["states/articulation/robot/joint_position"])[r - 1]

    keep = {}
    for i, m in enumerate(ds.sample_meta):
        if m[2] + hist - 1 == want[(m[0], m[1])]:
            keep[info[(m[0], m[1])]] = i
    order = sorted(keep.items(), key=lambda kv: kv[1])
    print(f"[{a.row}] {len(keep)} anchor windows (episodes in contract: {n_ep}); history_torque_lag "
          f"{getattr(cfg, 'history_torque_lag', 0)}")

    # against the same data cut with history 13 / lag 0 (rows t-12..t), the encoder torques must be tau[t-12..t-1]
    # for a lag-1 checkpoint (tau[t-11..t] for lag 0), the states x[t-11..t], and the first future torque tau_t
    lag = int(getattr(cfg, "history_torque_lag", 0) or 0)
    c13 = copy.deepcopy(cfg)
    c13.history_len, c13.history_torque_lag = hist + 1, 0
    ds13 = build_loader(c13, "val", a.batch_size, dataset_file)[0].dataset
    pos13 = {(m[0], m[1], m[2] + hist): j for j, m in enumerate(ds13.sample_meta)}
    for (b_, e_), i in order[:: max(1, len(order) // 50)] + [kv for kv in order if kv[0] == (0, 241)]:
        m = ds.sample_meta[i]
        j = pos13[(m[0], m[1], m[2] + hist - 1)]
        ht13, hs13 = ds13.history_torques[j], ds13.history_states[j]
        assert np.array_equal(ds.history_states[i], hs13[1:])
        assert np.array_equal(ds.future_torques[i][0], ht13[-1])
        assert np.array_equal(ds.history_torques[i], ht13[:-1] if lag == 1 else ht13[1:])
    print(f"[{a.row}] lag check passed (lag {lag}: encoder torques = tau[t-{11 + lag}..t-{lag}], future[0] = tau_t)")

    from torch.utils.data import DataLoader, Subset
    sub = DataLoader(Subset(ds, [kv[1] for kv in order]), batch_size=a.batch_size, shuffle=False, num_workers=0)
    out = {k: [] for k in ("hist_states", "hist_torques", "fut_states", "fut_torques", "pred_states", "mass")}
    model = loaded.model
    has_ctx = getattr(model, "use_context_encoder", False)
    with torch.inference_mode():
        for batch in sub:
            out["pred_states"].append(predict_batch(model, batch, dev, H).float().cpu().numpy())
            out["hist_states"].append(batch["history_states"].numpy())
            out["hist_torques"].append(batch["history_torques"].numpy()[..., :dof])
            out["fut_states"].append(batch["future_states"].numpy())
            out["fut_torques"].append(batch["future_torques"].numpy()[..., :dof])
            out["mass"].append(batch["object_context"].numpy()[:, 0])
    R = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    keys = [kv[0] for kv in order]
    pos = {k: i for i, k in enumerate(keys)}
    R["pred_frozen"] = R["pred_states"].copy()
    if has_ctx:
        # context from the null branch's window (identical states; its anchor torque is the null one)
        src = np.array([pos.get((0, e), pos[(b, e)]) for b, e in keys])
        with torch.inference_mode():
            for s0 in range(0, len(keys), a.batch_size):
                sl = slice(s0, s0 + a.batch_size)
                hs = torch.from_numpy(R["hist_states"][sl]).to(dev)
                hsn = torch.from_numpy(R["hist_states"][src[sl]]).to(dev)
                htn = torch.from_numpy(np.asarray([ds.history_torques[keep[keys[j]]] for j in src[sl]])).to(dev)
                ht = torch.from_numpy(np.asarray([ds.history_torques[keep[keys[j]]] for j in range(len(keys))[sl]])).to(dev)
                ft = torch.from_numpy(np.asarray([ds.future_torques[keep[keys[j]]] for j in range(len(keys))[sl]])).to(dev)
                ctx = model.encode_context(hsn, htn, deterministic_context=True)
                model.encode_context = lambda *_a, **_k: ctx
                try:
                    R["pred_frozen"][sl] = model(hs, ft, None, history_torques=ht, return_aux=False).float().cpu().numpy()
                finally:
                    del model.encode_context
        # sanity: on the null branches the frozen prediction IS the native one
        nb = np.array([b == 0 for b, _ in keys])
        print("frozen == native on null branches: max |diff|",
              float(np.abs(R["pred_frozen"][nb] - R["pred_states"][nb]).max()))
    R["branch"] = np.array([b for b, _ in keys])
    R["env"] = np.array([e for _, e in keys])
    R["slip_m"] = np.array([flags[k][0] for k in keys])
    R["fell"] = np.array([flags[k][1] for k in keys])
    R["wall"] = np.array([flags[k][2] for k in keys])
    R["q_abs_anchor"] = np.stack([qabs[k] for k in keys]).astype(np.float32)
    # a fork pair must be the same episode up to the anchor
    null = {e: i for i, (b, e) in enumerate(keys) if b == 0}
    for i, (b, e) in enumerate(keys):
        if b > 0 and e in null:
            assert np.abs(R["hist_states"][i][:, :2 * dof] - R["hist_states"][null[e]][:, :2 * dof]).max() < 1e-6
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **R, row=a.row, ckpt=a.ckpt, dataset=ID, dt=float(cfg.dt), dof=dof)
    print("->", a.out, {k: v.shape for k, v in R.items()})


if __name__ == "__main__":
    main()
