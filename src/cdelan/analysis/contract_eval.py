"""Window loader for a contract split (``build_loader``) and per-anchor scoring helpers.

Every number is a join over per-window records (episode, mass, object-position error at every horizon step), so an
aggregate can be re-derived without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cdelan.data import object_mlp_dataset
from cdelan.eval.rollout import load_checkpoint_model
from cdelan.analysis.predict import predict_batch

BOOTSTRAP_RESAMPLES = 1000


def build_loader(cfg, split: str, batch_size: int, dataset_file: str | None):
    from cdelan.eval.rollout import _resolve_hdf5_paths

    paths = _resolve_hdf5_paths(
        dataset_file=dataset_file if dataset_file is not None else cfg.dataset_file,
        dataset_id=None if dataset_file is not None else cfg.dataset_id,
        dataset_dir=cfg.dataset_dir,
    )
    refs = object_mlp_dataset.load_all_episode_refs(paths)
    refs = object_mlp_dataset.resolve_split_refs(refs, cfg, split)
    layout = object_mlp_dataset.RobotObjectWMStateLayout(
        robot_dof=cfg.robot_dof, action_dim=cfg.action_dim, torque_dim=cfg.torque_dim
    )
    ds = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=refs,
        history_len=cfg.history_len,
        rollout_horizon=cfg.rollout_horizon,
        dt=cfg.dt,
        torque_key=cfg.torque_key,
        filter_pre_contact=cfg.filter_pre_contact,
        object_displacement_threshold=cfg.object_displacement_threshold,
        object_velocity_threshold=cfg.object_velocity_threshold,
        contact_consecutive_steps=cfg.contact_consecutive_steps,
        contact_settle_steps=cfg.contact_settle_steps,
        subtract_env_origin=cfg.subtract_env_origin,
        phase_filter=cfg.phase_filter,
        # The checkpoint's own clock. Omitted, this defaults to False and an fd_clock run is
        # scored on the logged qd/qdd it was never trained on -- a train/eval mismatch that is
        # silent, because the object-position TARGET is the same either way and only the
        # rollout's inputs and initial state change. No-op for every fd_clock=False run.
        fd_clock=cfg.fd_clock,
        context_label_key=cfg.context_label_key,
        recompute_object_ang_vel=cfg.recompute_object_ang_vel,
        align_object_clock=cfg.align_object_clock,
        history_torque_lag=int(getattr(cfg, "history_torque_lag", 0) or 0),
        layout=layout,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, len(refs), paths


@torch.inference_mode()
def collect_records(model, loader, device, robot_dof: int, horizon: int):
    """Per-window object-position error [cm] at every horizon step, plus mass and episode."""
    o = 2 * robot_dof
    errs, masses, episodes, latents = [], [], [], []
    n_nonfinite = 0
    for batch in loader:
        pred = predict_batch(model, batch, device, horizon).float().cpu().numpy()
        gt = batch["future_states"].numpy()
        ok = np.isfinite(pred).reshape(len(pred), -1).all(axis=1)
        n_nonfinite += int((~ok).sum())
        if not ok.any():
            continue
        d = pred[ok, :, o : o + 3] - gt[ok, :, o : o + 3]
        errs.append(np.linalg.norm(d, axis=-1) * 100.0)  # (B, horizon) cm
        masses.append(batch["object_context"][:, 0].numpy()[ok])
        episodes.append(batch["episode_id"].numpy()[ok])
        # The latent itself, not the mass head. `lambda_mass_aux` is 0 in these runs, so the
        # head is initialised and never trained -- reading it would report an untrained
        # network's output as an identification error. z is saved instead and identification
        # is measured downstream by a probe fitted on the train band, which is exactly how
        # the contract treats a row whose context never commits to a scalar.
        if getattr(model, "use_context_encoder", False):
            ctx = model.encode_context(
                batch["history_states"].to(device),
                batch["history_torques"].to(device),
                deterministic_context=True,
            )
            if ctx.mu is not None:
                latents.append(ctx.mu.float().cpu().numpy()[ok])
    out = {
        "error_cm": np.concatenate(errs, axis=0),
        "mass_kg": np.concatenate(masses, axis=0),
        "episode_id": np.concatenate(episodes, axis=0),
        "n_nonfinite": n_nonfinite,
    }
    if latents:
        out["z"] = np.concatenate(latents, axis=0)
    return out


def episode_bootstrap_ci(values: np.ndarray, episodes: np.ndarray, seed: int = 0):
    """CI95 of the mean, resampling whole episodes with replacement."""
    uniq = np.unique(episodes)
    by_ep = {e: values[episodes == e] for e in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        means[i] = np.concatenate([by_ep[e] for e in draw]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mass_binned(errors_h8: np.ndarray, mass: np.ndarray, width: float, min_anchors: int):
    """Mean h=8 error per mass bin -- the shape the contract's extrapolation figure uses."""
    lo = np.floor(mass.min() / width) * width
    hi = np.ceil(mass.max() / width) * width
    edges = np.arange(lo, hi + width / 2, width)
    centres, means, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (mass >= a) & (mass < b)
        if sel.sum() < min_anchors:
            continue
        centres.append(float((a + b) / 2))
        means.append(float(errors_h8[sel].mean()))
        counts.append(int(sel.sum()))
    return centres, means, counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", help="val | train | any key of external_split_json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None, help="record filename stem (default: <ckpt dir>_<split>)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--bin_width_kg", type=float, default=0.25)
    ap.add_argument("--min_anchors", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    loaded = load_checkpoint_model(args.ckpt, device=device)
    cfg = loaded.config
    horizon = cfg.rollout_horizon
    loader, n_episodes, paths = build_loader(cfg, args.split, args.batch_size, args.dataset)
    tag = args.tag or f"{Path(args.ckpt).parent.name}_{args.split}"
    print(f"[contract_eval] {tag}: {n_episodes} episodes, horizon {horizon}, device {args.device}")

    rec = collect_records(loaded.model, loader, device, cfg.robot_dof, horizon)
    err, mass, eps = rec["error_cm"], rec["mass_kg"], rec["episode_id"]
    h8 = err[:, horizon - 1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = out_dir / f"records_{tag}.npz"
    np.savez_compressed(records, **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)})

    ci_lo, ci_hi = episode_bootstrap_ci(h8, eps)
    centres, means, counts = mass_binned(h8, mass, args.bin_width_kg, args.min_anchors)
    summary = {
        "checkpoint": os.path.abspath(args.ckpt),
        "split": args.split,
        "dataset_paths": paths,
        "n_episodes": n_episodes,
        "n_windows": int(len(h8)),
        "n_nonfinite": rec["n_nonfinite"],
        "horizon": horizon,
        "seed": cfg.seed,
        "mass_kg": {"min": float(mass.min()), "max": float(mass.max()), "mean": float(mass.mean())},
        "obj_pos_cm": {
            "per_horizon": [float(x) for x in err.mean(axis=0)],
            "h8": float(h8.mean()),
            "h8_median": float(np.median(h8)),
            "h8_ci95": [ci_lo, ci_hi],
        },
        "extrapolation": {
            "bin_width_kg": args.bin_width_kg,
            "min_anchors": args.min_anchors,
            "mass_kg": centres,
            "error_cm": means,
            "n_anchors": counts,
        },
        "records_file": records.name,
    }
    if "z" in rec:
        # Reported here only as a within-band association. The probe that turns z into a
        # kilogram estimate is fitted on the TRAIN band and applied here; fitting it on the
        # band it scores would report how well the band can be memorised, not identified.
        summary["latent"] = {
            "dim": int(rec["z"].shape[1]),
            "corr_with_mass_per_dim": [
                float(np.corrcoef(rec["z"][:, j], mass)[0, 1]) for j in range(rec["z"].shape[1])
            ],
        }

    path = out_dir / f"summary_{tag}.json"
    path.write_text(json.dumps(summary, indent=1))
    print(f"[contract_eval] windows={summary['n_windows']}  "
          f"obj_pos h8 = {summary['obj_pos_cm']['h8']:.3f} cm "
          f"[{ci_lo:.3f}, {ci_hi:.3f}]")
    if "latent" in summary:
        corrs = ", ".join(f"{c:+.3f}" for c in summary["latent"]["corr_with_mass_per_dim"])
        print(f"[contract_eval] z-mass corr per dim: {corrs}")
    print(f"[contract_eval] wrote {records}\n[contract_eval] wrote {path}")


if __name__ == "__main__":
    main()
