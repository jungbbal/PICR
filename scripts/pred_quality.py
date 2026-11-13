"""Open-loop prediction quality of one checkpoint on one evaluation set.

Every anchor window (12 history frames, all inside the carry phase) is rolled out autoregressively for
``--horizon`` 20 ms steps from the logged state and logged torques. Reported per anchor: ADD of the cube
(forward kinematics of the predicted arm pose + the cube rigidly attached to the hand) and per-joint angle
errors; summaries by payload mass and by object motion are written to ``--out``.

    python3 scripts/pred_quality.py --ckpt checkpoints/picr.pt \
        --protocol data/contracts/official_carry.json \
        --dataset_file data/raw/official_carry/official_carry.hdf5 --horizon 15 --out results/picr_carry_h15.json

PICR's encoder reads the history torques one frame late (``history_torque_lag 1``), so its anchors need rows
t-12..t+15 inside the carry phase. ``--anchor_history N`` scores a model with a shorter history on those same anchors.
The ``*_h8_*`` keys of the output hold the LAST step of ``--horizon``.
"""
import argparse
import json

import numpy as np
import torch

from cdelan.analysis.contract_eval import build_loader
from cdelan.analysis.predict import predict_batch
from cdelan.analysis.score_ckpt import ADD_FRACTION, ADD_POINTS, PAYLOAD_SIDE_M, _pose
from cdelan.eval.pose_metrics import add, cube_model
from cdelan.eval.rollout import load_checkpoint_model

MOVE_BINS_CM = [0.0, 0.5, 2.0, 5.0, np.inf]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset_file", default=None, help="score on this hdf5 instead of the checkpoint's dataset")
    ap.add_argument("--anchor_history", type=int, default=None,
                    help="window history used to pick the anchors (>= the checkpoint's history_len)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="score a different prediction horizon than the checkpoint's rollout_horizon;"
                         "the *_h8_* keys then hold the LAST step of this horizon")
    a = ap.parse_args()

    dev = torch.device("cuda")
    loaded = load_checkpoint_model(a.ckpt, device=dev)
    cfg = loaded.config
    cfg.external_split_json, cfg.external_split_val_key = a.protocol, a.split
    model_hist = cfg.history_len
    if a.anchor_history is not None:
        if a.anchor_history < model_hist:
            raise ValueError(f"--anchor_history {a.anchor_history} is shorter than the model's history {model_hist}.")
        cfg.history_len = a.anchor_history
    if a.horizon is not None:
        cfg.rollout_horizon = int(a.horizon)
    loader, n_ep, _ = build_loader(cfg, "val", 512, a.dataset_file)
    dof, H = cfg.robot_dof, cfg.rollout_horizon
    payload = cube_model("payload", PAYLOAD_SIDE_M, ADD_POINTS, symmetric=False)
    thr = ADD_FRACTION * payload.diameter

    err_m, err_h, add_m, add_h, move, mass, qerr = [], [], [], [], [], [], []
    for batch in loader:
        if a.anchor_history is not None:
            batch["history_states"] = batch["history_states"][:, -model_hist:]
            batch["history_torques"] = batch["history_torques"][:, -model_hist:]
        pred = predict_batch(loaded.model, batch, dev, H).float().cpu().numpy()
        gt = batch["future_states"].numpy()
        last = batch["history_states"].numpy()[:, -1:]
        hold = np.repeat(last, H, axis=1)
        q_g, t_g = _pose(gt, dof)
        q_p, t_p = _pose(pred, dof)
        q_h, t_h = _pose(hold, dof)
        err_m.append(np.linalg.norm(t_g - t_p, axis=-1))
        err_h.append(np.linalg.norm(t_g - t_h, axis=-1))
        add_m.append(add((q_g[:, -1], t_g[:, -1]), (q_p[:, -1], t_p[:, -1]), payload))
        add_h.append(add((q_g[:, -1], t_g[:, -1]), (q_h[:, -1], t_h[:, -1]), payload))
        move.append(np.linalg.norm(t_g[:, -1] - t_h[:, -1], axis=-1))
        mass.append(batch["object_context"][:, 0].numpy())
        qerr.append(np.abs(pred[:, :, :dof] - gt[:, :, :dof]))
    err_m, err_h = np.concatenate(err_m) * 100, np.concatenate(err_h) * 100
    add_m, add_h = np.concatenate(add_m), np.concatenate(add_h)
    move, mass = np.concatenate(move) * 100, np.concatenate(mass)
    qerr = np.degrees(np.concatenate(qerr))                       # (N, H, dof)

    def row(sel):
        return {
            "n": int(sel.sum()),
            "model_pos_cm_h8_mean": float(err_m[sel, -1].mean()),
            "model_pos_cm_h8_median": float(np.median(err_m[sel, -1])),
            "hold_pos_cm_h8_mean": float(err_h[sel, -1].mean()),
            "model_add01d_rate": float((add_m[sel] <= thr).mean()),
            "model_add02d_rate": float((add_m[sel] <= 2 * thr).mean()),   # ADD-0.2d (user report metric)
            "model_add_cm_h8_mean": float(add_m[sel].mean() * 100),
            "model_joint_deg_h8_mean": qerr[sel, -1].mean(0).tolist(),
            "model_joint_deg_h8_rmse": np.sqrt((qerr[sel, -1] ** 2).mean(0)).tolist(),
            "model_joint_deg2_h8_mse": (qerr[sel, -1] ** 2).mean(0).tolist(),
            "hold_add01d_rate": float((add_h[sel] <= thr).mean()),
            "hold_add02d_rate": float((add_h[sel] <= 2 * thr).mean()),
        }

    out = {
        "checkpoint": a.ckpt, "split": a.split, "n_episodes": n_ep, "robot_dof": dof, "horizon": H,
        "all": row(np.ones_like(move, bool)),
        "model_pos_cm_by_h": err_m.mean(0).tolist(),
        "hold_pos_cm_by_h": err_h.mean(0).tolist(),
        "model_joint_deg_by_h_mean": qerr.mean(0).tolist(),
        "dataset_file": a.dataset_file,
        "model_history_len": model_hist, "anchor_history": a.anchor_history,
        "by_motion_cm": {f"{lo}-{hi}": row((move >= lo) & (move < hi))
                          for lo, hi in zip(MOVE_BINS_CM[:-1], MOVE_BINS_CM[1:])},
        "by_mass_kg": {f"{m:.3f}": row(np.isclose(mass, m, atol=1e-3))
                        for m in np.unique(np.round(mass, 3))},
    }
    json.dump(out, open(a.out, "w"), indent=1)
    r = out["all"]
    print(f"[{dof}-DOF] h8 ADD mean {r['model_add_cm_h8_mean']:.3f} cm, ADD-0.1d {r['model_add01d_rate']:.3f}, joint mean [deg] "
          + " ".join(f"{x:.3f}" for x in r["model_joint_deg_h8_mean"]))
    print(f"[{dof}-DOF] h8 pos mean {r['model_pos_cm_h8_mean']:.3f} cm (hold {r['hold_pos_cm_h8_mean']:.3f}), "
          f"ADD-0.1d {r['model_add01d_rate']:.3f} (hold {r['hold_add01d_rate']:.3f})")
    for k, v in out["by_motion_cm"].items():
        print(f"  move {k:>8s} cm  n={v['n']:6d}  model {v['model_pos_cm_h8_mean']:.3f}  hold {v['hold_pos_cm_h8_mean']:.3f}"
              f"  ADD {v['model_add01d_rate']:.3f}/{v['hold_add01d_rate']:.3f}")


if __name__ == "__main__":
    main()
