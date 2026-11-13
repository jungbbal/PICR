"""Shared loading helpers for the latent-space figure scripts.

These used to live in ``scripts/compare_film_modulation.py``, a FiLM-ablation driver that
the figure scripts imported for its two helpers. That coupling dragged the 15-DOF model and
a set of hard-coded ablation checkpoint paths into every latent figure. Both helpers are
structure-agnostic, so they belong here on their own.
"""

from __future__ import annotations

import numpy as np
import torch

from cdelan import paths
from cdelan.data import object_mlp_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: Experiment 01's dataset. Kept as the fallback for checkpoints too old to name a
#: ``dataset_id``; anything newer should go through :func:`resolve_dataset_files`, which
#: follows the checkpoint instead of assuming this one.
try:
    TRAIN_FILE = str(paths.dataset_file("lift_randmass_phase3_9590ep"))
except FileNotFoundError:
    # Removed from raw/ in the cleanup. Only the legacy scripts that import TRAIN_FILE
    # (mass_rsa, z_drop_per_mass, pca_episode, z_converge_video) need it; everything else must still import.
    TRAIN_FILE = None


def resolve_dataset_files(cfg: dict) -> list[str]:
    """The hdf5 files the checkpoint was actually trained on.

    The trainer's own resolution, so a figure is always drawn on the dataset the checkpoint
    names rather than on a module-level constant. This is the same path
    ``context_z_latent.build_val_dataset`` takes; it lives here so the scripts that build
    their dataset by hand (the animation ones) can follow the checkpoint too.
    """
    import dataclasses
    import os
    import yaml
    from cdelan.training.train import TrainConfig, resolve_hdf5_paths

    # Checkpoints predating a config field would fail TrainConfig construction; backfill
    # from the default config exactly as build_val_dataset does.
    defaults_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "configs", "train_config.yaml")
    with open(defaults_path) as fh:
        merged = yaml.safe_load(fh) or {}
    merged.update(cfg)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    return list(resolve_hdf5_paths(TrainConfig(**{k: v for k, v in merged.items() if k in fields})))


def ridge_dir(z: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Least-squares direction in latent space that best predicts ``y``.

    Ridge rather than a plain solve because the latent's coordinates are correlated, which
    leaves the normal equations near-singular and the direction unstable run to run.
    """
    zc = z - z.mean(0, keepdims=True)
    a = zc.T @ zc + lam * np.eye(z.shape[1]) * max(np.trace(zc.T @ zc) / z.shape[1], 1e-12)
    return np.linalg.solve(a, zc.T @ (y - y.mean()))


def load_model(ckpt: str):
    """Rebuild a model from its checkpoint and return ``(model, config)``.

    Dispatches on ``config["model_type"]`` through the training-time builder, so it does not
    need to know which structure wrote the checkpoint.
    """
    from cdelan.eval.rollout import load_checkpoint_model

    loaded = load_checkpoint_model(ckpt, device=DEVICE)
    cfg = dict(torch.load(ckpt, map_location="cpu", weights_only=False)["config"])
    # Fields added after some checkpoints were written; keep readers total.
    cfg.setdefault("recompute_object_ang_vel", False)
    cfg.setdefault("align_object_clock", False)
    return loaded.model.eval(), cfg


def build_dataset(refs, cfg):
    """The rollout dataset exactly as the checkpoint's own config specified it.

    The layout has to come from the config too. It defaults to the 9-DOF arm-plus-fingers
    recording, so a 7-DOF checkpoint (fingers removed from the generalized coordinates)
    silently got 9-wide torques here and died inside the context encoder with a
    ``mat1 and mat2 shapes cannot be multiplied (256x552 and 504x256)`` -- 24 x (14 + 9)
    against the 24 x (14 + 7) it was built for. For a 9-DOF checkpoint these are the
    defaults and nothing changes.
    """
    layout = object_mlp_dataset.RobotObjectWMStateLayout(
        robot_dof=int(cfg["robot_dof"]),
        action_dim=int(cfg["action_dim"]),
        torque_dim=int(cfg["torque_dim"]),
    )
    return object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=refs,
        layout=layout,
        history_len=cfg["history_len"],
        rollout_horizon=cfg["rollout_horizon"],
        dt=cfg["dt"],
        torque_key=cfg["torque_key"],
        filter_pre_contact=cfg["filter_pre_contact"],
        object_displacement_threshold=cfg["object_displacement_threshold"],
        object_velocity_threshold=cfg["object_velocity_threshold"],
        contact_consecutive_steps=cfg["contact_consecutive_steps"],
        contact_settle_steps=cfg["contact_settle_steps"],
        subtract_env_origin=cfg["subtract_env_origin"],
        recompute_object_ang_vel=bool(cfg.get("recompute_object_ang_vel") or False),
        align_object_clock=bool(cfg.get("align_object_clock") or False),
        phase_filter=cfg["phase_filter"],
        window_stride=int(cfg.get("window_stride") or 1),
        history_torque_lag=int(cfg.get("history_torque_lag") or 0),
    )


def find_phase3_start(phase: np.ndarray, history_len: int, phase_value: int = 3) -> int | None:
    """First timestep whose preceding ``history_len`` window is entirely phase-3.

    Starting a rollout anywhere else feeds the model a history window that straddles the lift
    transition -- input it never saw under ``phase_filter: 3``. That has already produced one
    silently wrong result in this project (a rollout diverging from 196 mm to 1014 mm), so the
    start index is chosen rather than assumed.
    """
    total = len(phase)
    for t in range(history_len - 1, total - 1):
        if np.all(phase[t - history_len + 1 : t + 1] == phase_value):
            return t
    idx = np.where(phase == phase_value)[0]          # fallback: first phase-3 frame with history
    if len(idx):
        return int(max(idx[0], history_len - 1))
    return None
