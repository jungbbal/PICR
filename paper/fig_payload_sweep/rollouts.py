"""Per-anchor h = 15 open-loop rollouts of PICR on the figure-eight payload sweep (ramp8_up), cached for the
Table 1 ADD column.

Anchors: rows t-12..t of the window lie inside the carry phase (history_torque_lag 1: encoder torques tau[t-12..t-1],
states x[t-11..t]). Same loader, predict_batch and pose / ADD path as scripts/pred_quality.py.

    python3 paper/fig_payload_sweep/rollouts.py [--ckpt checkpoints/picr.pt] [--out <dir>]

Writes <out>/picr_ramp8up_h15.npz: episode, t (anchor frame), add_cm (N x H), gt/pred cube pose (pos N x H x 3,
R N x H x 3 x 3), anchor_pos (N x 3, cube at the anchor), mass_kg (N, true payload at the anchor).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from cdelan.analysis.predict import predict_batch
from cdelan.analysis.contract_eval import build_loader
from cdelan.analysis.score_ckpt import ADD_POINTS, PAYLOAD_SIDE_M, _pose
from cdelan.eval.pose_metrics import add, cube_model, to_Rt
from cdelan.eval.rollout import load_checkpoint_model

DATASET = "ramp8_up"


def as_Rt(q, t):
    q, t = np.asarray(q, np.float64), np.asarray(t, np.float64)
    lead = q.shape[:-1]
    R, T = to_Rt((q.reshape(-1, 4), t.reshape(-1, 3)))
    return R.reshape(*lead, 3, 3), T.reshape(*lead, 3)


@torch.inference_mode()
def run(row, ckpt, out_dir, H, dev):
    out = Path(out_dir) / f"{row}_ramp8up_h{H}.npz"
    loaded = load_checkpoint_model(ckpt, device=dev)
    cfg = loaded.config
    cfg.external_split_json = os.path.join(os.environ["PICR_CONTRACTS"], f"{DATASET}.json")
    cfg.external_split_val_key = "val"
    cfg.rollout_horizon = H
    hdf = os.path.join(os.environ["PICR_DATA_ROOT"], "raw", DATASET, f"{DATASET}.hdf5")
    lag = int(getattr(cfg, "history_torque_lag", 0) or 0)
    loader, n_ep, _ = build_loader(cfg, "val", 512, hdf)
    ds, dof = loader.dataset, cfg.robot_dof
    print(f"[{row}] {ckpt}: history_torque_lag {lag}, history {ds.history_len}, {len(ds)} windows")
    payload = cube_model("payload", PAYLOAD_SIDE_M, ADD_POINTS, symmetric=False)
    acc = {k: [] for k in ("gt_R", "gt_pos", "pred_R", "pred_pos", "anchor_pos", "mass_kg")}
    for batch in loader:
        pred = predict_batch(loaded.model, batch, dev, H).float().cpu().numpy()
        for key, src in (("gt", batch["future_states"].numpy()), ("pred", pred)):
            R, T = as_Rt(*_pose(src, dof))
            acc[f"{key}_R"].append(R.astype(np.float32)); acc[f"{key}_pos"].append(T.astype(np.float32))
        acc["mass_kg"].append(batch["object_context"][:, 0].numpy().astype(np.float32))
        acc["anchor_pos"].append(as_Rt(*_pose(batch["history_states"].numpy()[:, -1], dof))[1].astype(np.float32))
    rec = {k: np.concatenate(v) for k, v in acc.items()}
    N = len(rec["gt_pos"]); add_cm = np.full((N, H), np.nan)
    for h in range(H):
        ok = np.isfinite(rec["pred_pos"][:, h]).all(1) & np.isfinite(rec["pred_R"][:, h]).reshape(N, -1).all(1)
        add_cm[ok, h] = add((rec["gt_R"][ok, h].astype(np.float64), rec["gt_pos"][ok, h].astype(np.float64)),
                            (rec["pred_R"][ok, h].astype(np.float64), rec["pred_pos"][ok, h].astype(np.float64)),
                            payload) * 100.0
    episode = np.array([ds.sample_meta[j][1] for j in range(N)])
    t = np.array([int(ds.window_start[j]) + ds.history_len - 1 for j in range(N)])
    rec.update(episode=episode, t=t, add_cm=add_cm)
    N, add_cm = len(rec["t"]), rec["add_cm"]
    print(f"[{row}] {n_ep} episodes, {N} anchors; h{H} ADD mean {np.nanmean(add_cm[:, -1]):.4f} cm, "
          f"non-finite {int((~np.isfinite(add_cm[:, -1])).sum())}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{k: rec[k] for k in ("episode", "t", "add_cm", "gt_R", "gt_pos", "pred_R", "pred_pos",
                                                    "anchor_pos", "mass_kg")})
    tmp.rename(out)
    print(f"-> {out}")


def main():
    root = os.environ["PICR_ROOT"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(root, "checkpoints", "picr.pt"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("PICR_WORK", os.path.join(root, "work")),
                                                  "fig_payload_sweep", "cache"))
    ap.add_argument("--horizon", type=int, default=15)
    a = ap.parse_args()
    run("picr", a.ckpt, a.out, a.horizon, torch.device("cuda"))


if __name__ == "__main__":
    main()
