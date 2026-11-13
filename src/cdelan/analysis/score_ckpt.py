"""Score one checkpoint on a contract split: open-loop rollout of every anchor window, cube pose from forward
kinematics of the predicted arm, ADD of a 2000-point model of the 4.8 cm cube.

ADD is ``symmetric=False`` and the ADD-0.1d gate is 0.1 x the cube model's measured diameter
(Hinterstoisser et al., ACCV 2012), never a hardcoded length.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cdelan.analysis.contract_eval import build_loader
from cdelan.analysis.predict import predict_batch
from cdelan.eval.pose_metrics import add, cube_model, rotation_error_deg
from cdelan.eval.rollout import load_checkpoint_model

PAYLOAD_SIDE_M = 0.048
"""The lift cube's edge, from every dataset manifest."""

ADD_POINTS = 2000
ADD_FRACTION = 0.1
"""The `0.1` in ADD-0.1d. The threshold is 0.1 x the model's own measured diameter."""


def _pose(states: np.ndarray, dof: int):
    """(quat wxyz, translation m) for the object channels of a (..., state_dim) block."""
    o = 2 * dof
    return states[..., o + 3 : o + 7], states[..., o : o + 3]


@torch.inference_mode()
def score(model, loader, device, dof: int, horizon: int) -> dict:
    """Anchor means of position L2, ADD and rotation at every horizon step."""
    payload = cube_model("payload", PAYLOAD_SIDE_M, ADD_POINTS, symmetric=False)
    thresh_m = ADD_FRACTION * payload.diameter

    pos_sum = np.zeros(horizon)
    add_sum = np.zeros(horizon)
    rot_sum = np.zeros(horizon)
    add_pass = np.zeros(horizon, dtype=np.int64)
    add_final: list[np.ndarray] = []
    n = 0
    n_nonfinite = 0

    for batch in loader:
        pred = predict_batch(model, batch, device, horizon).float().cpu().numpy()
        gt = batch["future_states"].numpy()

        # Drop the whole anchor, not the offending step: a rollout that went
        # non-finite at h=5 has no meaningful h=6..8 either.
        ok = np.isfinite(pred).reshape(len(pred), -1).all(axis=1)
        n_nonfinite += int((~ok).sum())
        if not ok.any():
            continue
        pred, gt = pred[ok], gt[ok]

        q_p, t_p = _pose(pred, dof)
        q_g, t_g = _pose(gt, dof)

        pos_sum += np.linalg.norm(t_g - t_p, axis=-1).sum(axis=0)
        rot_sum += rotation_error_deg((q_g, t_g), (q_p, t_p)).sum(axis=0)

        # Per horizon step: ADD materialises (batch, n_points, 3), so stepping the
        # horizon keeps the peak allocation at one step's worth rather than eight.
        for h in range(horizon):
            d = add((q_g[:, h], t_g[:, h]), (q_p[:, h], t_p[:, h]), payload)
            add_sum[h] += d.sum()
            add_pass[h] += int((d <= thresh_m).sum())
            if h == horizon - 1:
                # Kept so --check can say WHY a pass count differs. ADD-0.1d is a
                # discrete gate, so an anchor sitting a few ulp from it flips on
                # float noise the continuous means never show. 1.2 MB at this size.
                add_final.append(d.copy())

        n += len(pred)

    if n == 0:
        raise SystemExit("every anchor was non-finite -- nothing to score")

    cm = 100.0
    return {
        "n_anchors": n,
        "n_anchors_nonfinite": n_nonfinite,
        "position_cm_by_h": (pos_sum / n * cm).tolist(),
        "add_cm_by_h": (add_sum / n * cm).tolist(),
        "rotation_deg_by_h": (rot_sum / n).tolist(),
        "add_01d_pass": int(add_pass[horizon - 1]),
        "add_01d_rate": float(add_pass[horizon - 1] / n),
        "add_01d_threshold_cm": float(thresh_m * cm),
        "_add_final_m": np.concatenate(add_final),
        "_threshold_m": thresh_m,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--protocol", required=True, help="the contract JSON; its `splits` win")
    ap.add_argument("--split", default="val", help="any key of the contract's splits")
    ap.add_argument("--out", default=None, help="manifest path (default: stdout only)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", default=None,
                    help="an existing manifest to diff against; exits non-zero on mismatch")
    a = ap.parse_args(argv)

    device = torch.device(a.device)
    loaded = load_checkpoint_model(a.ckpt, device=device)
    cfg = loaded.config
    horizon = cfg.rollout_horizon

    # train_protocol passed the contract on the command line and the saved config did
    # not keep it (external_split_json is "" in every shipped checkpoint), so the band
    # is named here explicitly rather than inferred from the checkpoint.
    cfg.external_split_json = a.protocol
    cfg.external_split_val_key = a.split
    loader, n_episodes, _paths = build_loader(cfg, "val", a.batch_size, None)
    print(f"[score_ckpt] {a.split}: {n_episodes} episodes, horizon {horizon}, {a.device}",
          flush=True)

    agg = score(loaded.model, loader, device, cfg.robot_dof, horizon)
    manifest = {
        "checkpoint": str(Path(a.ckpt).resolve()),
        "set": a.split,
        "epoch": int(getattr(loaded, "epoch", cfg.epochs)),
        "n_episodes": n_episodes,
        "n_anchors": agg["n_anchors"],
        "horizon": horizon,
        "position_cm": agg["position_cm_by_h"][-1],
        "add_cm": agg["add_cm_by_h"][-1],
        "rotation_deg": agg["rotation_deg_by_h"][-1],
        "position_cm_by_h": agg["position_cm_by_h"],
        "add_cm_by_h": agg["add_cm_by_h"],
        "rotation_deg_by_h": agg["rotation_deg_by_h"],
        "add_01d_pass": agg["add_01d_pass"],
        "add_01d_rate": agg["add_01d_rate"],
        "add_01d_threshold_cm": agg["add_01d_threshold_cm"],
    }
    if agg["n_anchors_nonfinite"]:
        manifest["n_anchors_nonfinite"] = agg["n_anchors_nonfinite"]

    print(f"  position {manifest['position_cm']:.15f} cm   add {manifest['add_cm']:.15f} cm")
    print(f"  rotation {manifest['rotation_deg']:.13f} deg  "
          f"ADD-0.1d {manifest['add_01d_pass']:,}/{manifest['n_anchors']:,}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(manifest, indent=1))
        print(f"  -> {a.out}")

    if a.check:
        ref = json.loads(Path(a.check).read_text())
        worst, bad = 0.0, []
        for k in ("position_cm", "add_cm", "rotation_deg", "add_01d_threshold_cm"):
            d = abs(manifest[k] - ref[k])
            worst = max(worst, d)
            print(f"  {k:22s} {manifest[k]!r:22.22} vs {ref[k]!r:22.22}  d={d:.3e}")
        for k in ("n_episodes", "n_anchors", "add_01d_pass"):
            same = manifest[k] == ref[k]
            print(f"  {k:22s} {manifest[k]:>10,} vs {ref[k]:>10,}  "
                  f"{'MATCH' if same else 'DIFFER'}")
            if not same:
                bad.append(k)

        # `n_episodes` and `n_anchors` differing means the wrong split or the wrong
        # dataset -- never tolerable. `add_01d_pass` is different in kind: it is a
        # discrete gate over a continuous quantity, so an anchor within float noise
        # of the threshold flips sides between two runs that agree everywhere else.
        # Explain that rather than assert it: count how many anchors are actually
        # close enough to account for the gap, and only forgive a gap that size.
        gate_ok = False
        if bad == ["add_01d_pass"]:
            d = np.abs(agg["_add_final_m"] - agg["_threshold_m"])
            gap = abs(manifest["add_01d_pass"] - ref["add_01d_pass"])
            straddlers = int((d < 1e-7).sum())
            nearest = float(d.min())
            print(f"  {'':22s} nearest anchor to the gate: {nearest:.3e} m; "
                  f"{straddlers} within 1e-7 m; gap is {gap}")
            gate_ok = gap <= straddlers and worst <= 1e-6
            if gate_ok:
                print(f"  {'':22s} -> boundary straddle, not a scoring difference")

        if (bad and not gate_ok) or not math.isfinite(worst) or worst > 1e-6:
            raise SystemExit(
                f"MISMATCH (counts differing: {bad or 'none'}, worst float {worst:.3e})")
        print(f"  reproduces {a.check} (worst float delta {worst:.3e}"
              + (f", {abs(manifest['add_01d_pass'] - ref['add_01d_pass'])} gate straddle)"
                 if gate_ok else ")"))


if __name__ == "__main__":
    main()
