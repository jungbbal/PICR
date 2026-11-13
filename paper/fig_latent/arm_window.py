"""Panel (b)'s example window: one 3 kg carry window rolled out open loop with the window's own z (adapted, "z_3kg")
and with z_stale ("z_0.1kg"), keeping the joint angles (for the Isaac replay) and the cube poses (for the drawn tracks).

    python3 arm_window.py --ckpt checkpoints/picr.pt --stale <work>/stale_z.json --episode demo_1195 --window 38 \
        --horizon 12 --out <work>/arm_window_h12.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cdelan import paths
from cdelan.analysis._common import DEVICE, build_dataset, load_model
from cdelan.analysis.z_rollout import CUBE_SIDE


@torch.no_grad()
def roll(model, hs, ft, z):
    """`ArmPayloadWMDynamics.forward` with z supplied. -> (q relative to the default pose, object pose)."""
    r = model.robot_dof
    state = hs[:, -1]
    grasp = model.infer_grasp(state)
    q, dq = state[:, :r], state[:, r : 2 * r]
    qs, obj = [], []
    for k in range(ft.shape[1]):
        q, dq, _ = model._step_arm(q, dq, ft[:, k], z=z)
        qs.append(q)
        obj.append(model.assemble_state(q, dq, grasp)[:, 2 * r : 2 * r + 7])
    return torch.stack(qs, 1), torch.stack(obj, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stale", required=True, help="stale_z.json")
    ap.add_argument("--ood-dataset", default="official_ood_2to4kg")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z_stale = np.asarray(json.loads(Path(args.stale).read_text())["z_stale"], dtype=np.float32)
    model, cfg = load_model(args.ckpt)
    cfg = dict(cfg, rollout_horizon=int(args.horizon))
    dt, r = float(cfg["dt"]), model.robot_dof

    ds = build_dataset([(str(paths.dataset_file(args.ood_dataset)), args.episode)], cfg)
    hit = [i for i, m in enumerate(ds.sample_meta) if int(m[2]) == args.window]
    if not hit:
        raise SystemExit(f"window {args.window} not in {args.episode}")
    j = hit[0]
    hs = torch.from_numpy(np.asarray(ds.history_states[j : j + 1])).to(DEVICE)
    ht = torch.from_numpy(np.asarray(ds.history_torques[j : j + 1])).to(DEVICE)
    ft = torch.from_numpy(np.asarray(ds.future_torques[j : j + 1])).to(DEVICE)
    with torch.no_grad():
        z = model.encode_context(hs, ht, deterministic_context=True).mu
    zs = torch.as_tensor(z_stale, dtype=z.dtype, device=DEVICE).expand_as(z)

    fut = np.asarray(ds.future_states[j])
    anc = np.asarray(ds.history_states[j])[-1]
    q_abs = model.default_joint_pos.detach().cpu().numpy()
    out = {"gt_q": fut[:, :r] + q_abs, "gt_obj": fut[:, 2 * r : 2 * r + 7],
           "anchor_q": anc[:r] + q_abs, "anchor_obj": anc[2 * r : 2 * r + 7],
           "hist_q": np.asarray(ds.history_states[j])[:, :r] + q_abs,
           "hist_obj": np.asarray(ds.history_states[j])[:, 2 * r : 2 * r + 7]}
    for key, zz in (("ours", z), ("stale", zs)):
        qr, obj = (t[0].cpu().numpy() for t in roll(model, hs, ft, zz))
        out[f"{key}_q"], out[f"{key}_obj"] = qr + q_abs, obj
        out[f"err_{key}"] = np.linalg.norm(obj[:, :3] - out["gt_obj"][:, :3], axis=-1) * 100.0

    np.savez_compressed(args.out, cube_side=CUBE_SIDE, dt=dt, robot_dof=r, episode=args.episode, window=args.window,
                        horizon=int(out["gt_q"].shape[0]), z_adapted=z[0].cpu().numpy(), z_stale=z_stale, **out)
    print(f"{args.episode} @ {args.window}: h={args.horizon} end-point error  adapted {out['err_ours'][-1]:.2f}  "
          f"stale {out['err_stale'][-1]:.2f} cm -> {args.out}")


if __name__ == "__main__":
    main()
