"""Rollout dataset for the DeLaN + object-MLP world model."""

from __future__ import annotations

from collections import defaultdict
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contact_labels import first_object_motion_timestep
from .dataset import (
    discover_hdf5_files,
    h5_open,
    load_all_episode_refs,
    load_episode_names,
    resolve_split_refs,
    split_episode_refs,
)
from .hdf5_schema import EpisodeRef, Hdf5Groups, RobotObjectStateLayout

# Which per-episode physical parameters the contrastive context loss is structured by, and
# how many numbers each axis carries. A multi-dimensional axis must list every parameter
# that was randomized: supervising one of them leaves the rest as unlabelled nuisance in the
# encoder's input, which is what made the RandDamping run uninterpretable.
CONTEXT_LABEL_DIMS = {
    "mass": 1,
    "arm_damping": 1,
    "friction_armature": 2,
    "com_offset": 1,
    "mass_com": 2,
    "mass_com3": 4,
    "inertial_pi": 10,
}
CONTEXT_LABEL_KEYS = frozenset(CONTEXT_LABEL_DIMS)

RobotObjectWMStateLayout = RobotObjectStateLayout


_MCG_ARM_DOF = 7

# |computed_torque - applied_torque| above this (N*m) means the actuator hit its effort
# limit. The logged difference is either exactly 0 or O(1-50) N*m, so the threshold is not
# a tuning knob.
_EFFORT_SATURATION_TOL = 0.01
"""Width of the GT_full sidecar: the Franka arm without the two finger joints, matching
``id_loss_arm_dof``. The fingers carry the grip's contact forces (rms |tau| 44 N*m against an
|Hqdd + c + g| of 0.24) and this model has no contact term, so their residual is unfixable."""


def _quat_to_body_ang_vel(quat: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity finite-differenced from a quaternion sequence.

    quat: (T, 4) Hamilton [w, x, y, z]. Returns (T, 3) omega such that
    integrate_quat(q[t], omega[t], dt) == q[t+1] (the exact log-map rate). The
    last step is padded with the previous value. This reconstructs the true
    (small) object rotation, unlike the jitter-contaminated logged angular vel.
    """
    q = quat.astype(np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    w, x, y, z = q[:-1, 0], q[:-1, 1], q[:-1, 2], q[:-1, 3]      # q[t]
    nw, nx, ny, nz = q[1:, 0], q[1:, 1], q[1:, 2], q[1:, 3]      # q[t+1]
    # dq = conj(q[t]) x q[t+1]  (body-frame delta rotation)
    dw = w * nw + x * nx + y * ny + z * nz
    dx = w * nx - x * nw - y * nz + z * ny
    dy = w * ny + x * nz - y * nw - z * nx
    dz = w * nz - x * ny + y * nx - z * nw
    vnorm = np.sqrt(dx * dx + dy * dy + dz * dz)
    angle = 2.0 * np.arctan2(vnorm, np.clip(dw, -1.0, 1.0))       # [0, 2pi)
    angle = np.where(angle > np.pi, angle - 2.0 * np.pi, angle)   # shortest arc
    scale = np.where(vnorm > 1e-8, angle / (np.maximum(vnorm, 1e-8) * dt), 0.0)
    omega = np.stack([dx * scale, dy * scale, dz * scale], axis=1)
    if len(omega) == 0:
        return np.zeros((quat.shape[0], 3), dtype=np.float32)
    return np.concatenate([omega, omega[-1:]], axis=0).astype(np.float32)


def _fd_clock_channels(joint_pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """`(qd, qdd)` derived from the logged positions alone. See `fd_clock`.

    Both are built on the OBS clock, which is what `joint_pos` is on, and then `qdd` is
    shifted onto the states clock that `robot_dynamics/qdd` uses -- `obs[i] == states[i-1]`
    exactly (verified on the recording: `joint_vel` matches bitwise at that shift and
    `joint_pos` to 1.6e-5 rad, the constant being the default pose). `align_object_clock`
    then takes `joint_pos[1:]` against `qdd[:-1]`, so after it index i carries
    `(q_s[i], qd_s[i], qdd_s[i])` and both

        qd_s[i+1] = qd_s[i] + qdd_s[i] dt        q_s[i+1] = q_s[i] + qd_s[i+1] dt

    hold as identities -- semi-implicit Euler reproduces the recorded trajectory exactly.
    A constant offset in `joint_pos` (obs is relative to the default pose, states absolute)
    cancels in every difference, so deriving from either array gives the same answer.

    The first row of `qd` and the last of `qdd` have no predecessor/successor and are
    filled from their neighbour; anchors never reach either end, because every window
    carries `history_len` frames of context.
    """
    qd = np.empty_like(joint_pos)
    qd[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
    qd[0] = qd[1]
    qdd_obs = np.empty_like(qd)
    qdd_obs[:-1] = (qd[1:] - qd[:-1]) / dt
    qdd_obs[-1] = qdd_obs[-2]
    qdd = np.empty_like(qdd_obs)  # qdd_states[i] = qdd_obs[i+1]
    qdd[:-1] = qdd_obs[1:]
    qdd[-1] = qdd_obs[-1]
    return qd, qdd


class RobotObjectWMRolloutDataset(Dataset):
    """Rollout samples with robot state, object state, torque, and object context."""

    def __init__(
        self,
        episode_refs: List[EpisodeRef],
        history_len: int,
        rollout_horizon: int,
        dt: float,
        torque_key: str = "applied_torque",
        filter_pre_contact: bool = False,
        object_displacement_threshold: float = 0.005,
        object_velocity_threshold: float = 0.02,
        contact_consecutive_steps: int = 3,
        contact_settle_steps: int = 5,
        subtract_env_origin: bool = True,
        layout: RobotObjectWMStateLayout | None = None,
        recompute_object_ang_vel: bool = False,
        phase_filter: int | None = None,
        align_object_clock: bool = False,
        context_label_key: str = "mass",
        mcg_target_file: str | None = None,
        filter_effort_saturated: bool = False,
        filter_velocity_saturated: bool = False,
        fd_clock: bool = False,
        window_stride: int = 1,
        history_torque_lag: int = 0,
    ) -> None:
        super().__init__()
        if history_len < 1:
            raise ValueError("history_len must be >= 1.")
        if rollout_horizon < 1:
            raise ValueError("rollout_horizon must be >= 1.")
        self.history_len = history_len
        self.rollout_horizon = rollout_horizon
        # Anchor every `window_stride`-th timestep instead of every one. The 0.5 ms (substep)
        # recordings have 40x the rows of the 20 ms ones, so stride 1 would build ~40x the
        # windows; a stride keeps the window count (and RAM) at the 20 ms datasets' scale
        # while each window still runs at the recording's own dt.
        if window_stride < 1:
            raise ValueError("window_stride must be >= 1.")
        self.window_stride = int(window_stride)
        # shift the HISTORY torque window back by this many rows. 0 (default) is the
        # original cut torques[t-k : t+1] next to state[t-k : t+1], whose last element tau_t is the first
        # future action. lag 1 gives torques[t-k-1 : t] = tau_{t-12..t-1}: the encoder sees only past
        # torques. The state window and the future torques torques[t : t+h] are unchanged; the anchor
        # grid is unchanged too, and an anchor is valid only if rows t-k-lag .. t+h pass the filters.
        if history_torque_lag < 0:
            raise ValueError("history_torque_lag must be >= 0.")
        self.history_torque_lag = int(history_torque_lag)
        self.dt = dt
        self.torque_key = torque_key
        self.subtract_env_origin = subtract_env_origin
        # The logged object angular velocity is contaminated by contact-solver jitter
        # (~10x the pose-implied rate, corr 0.25). When True, replace it with the
        # body-frame angular velocity finite-differenced from the (clean) quaternion
        # trajectory, so it is consistent with integrate_quat and actually integrable.
        self.recompute_object_ang_vel = bool(recompute_object_ang_vel)
        # Put the obs/* (robot, torque, action, phase) and states/* (object pose)
        # halves of the state vector on the same clock; see _build_samples.
        self.align_object_clock = bool(align_object_clock)
        # When set, only keep rollout windows whose every timestep (history + future)
        # has obs/phase == phase_filter. Used to train/eval on phase 3 (post-lift) only.
        self.phase_filter = None if phase_filter is None else int(phase_filter)
        # Drop windows containing a frame where any arm joint's actuator hit its effort
        # limit -- `computed_torque` (what the PD asked for) differs from `applied_torque`
        # (what was delivered). Measured on lift_masscom_lattice_phase3: j4/j5/j6 clip at
        # +-12 N*m in 27/5/38% of phase-3 frames while the PD demands 52-63, and in exactly
        # those frames tau - (H qdd + c + g) is 28 N*m against 0.3-1.1 N*m elsewhere -- with
        # GT_full, so it is not the payload (adding it moves 28.24 -> 28.54, and the residual
        # correlates 0.01 with payload mass). Fitting that unphysical half is what forces the
        # learned wrist inertia ~100x above the true value; see the notes on
        # inverse_dynamics_loss. 46.5% of 20-frame windows survive.
        self.filter_effort_saturated = bool(filter_effort_saturated)
        # Drop windows containing a frame where any arm joint sits on its VELOCITY wall
        # (|qd| > 0.99 * [2.175]*4 + [2.61]*3). There the solver's limit constraint absorbs
        # torque that no logged field carries, so `tau = H qdd + c + g` is false by
        # construction: measured on the 0.01 s pilot, the residual against GT arm terms and
        # the analytic payload is 14.42 N*m median on those frames and 0.011 N*m off them.
        # The flag is written by the dataset post-processing as obs/velocity_saturated.
        # It is not free: measured on the 448-episode pilot AFTER the success filter and the
        # trim, 32.4% of phase-3 frames are on the wall and only 59.6% of (k=24, h=16)
        # windows survive -- the wall binds exactly where the arm is moving, which is also
        # where the payload signal is (study 15 lost the payload by deleting the
        # effort-saturated frames). Which way that trade falls is an ablation, not a default.
        # Replace the logged velocity and acceleration by finite differences of the logged
        # POSITION, so that the three channels describe one trajectory.
        #
        # The recording's own `q` and `qd` do not: `(q[t+1] - q[t])/dt` disagrees with the
        # logged `qd[t+1]` by 21% rms even at 0.01 s, and no one-step rule closes it -- the
        # best least-squares blend `a dt qd_t + b dt qd_{t+1}` gives a = 0.299, b = 0.720
        # with 9% of every step still unexplained. Two causes are visible in that number:
        # the asymmetry is a genuine phase offset between the channels, and the magnitude at
        # the fast joints is ordinary discretisation (j6's 25.8 Hz against dt = 0.01 puts
        # `omega dt` at 1.6, so even a trapezoid rule carries ~22%). PhysX cannot report the
        # velocity that actually moved the joint -- TGS averages it over several internal
        # position iterations -- but it needs no solver internals: it IS the backward
        # difference.
        #
        # Measured cost of the mismatch, ADD at 0.16 s on lift_sub0p01 id_test: replaying the
        # true logged velocities lands at 1.036 cm (all anchors) / 0.920 (wall-clean) against
        # a 0.196 / 0.015 cm floor from FK and the frozen grasp alone. On this clock the
        # replay hits that floor exactly, by construction. It is not paid for in the dynamics
        # either: the ledger `tau - (M qdd + c + g + payload)` goes 0.901 -> 0.487 N*m rms
        # (wall-clean), i.e. the derived acceleration explains the recorded torque BETTER
        # than the logged one.
        #
        # `robot_dynamics/{coriolis, gravity}` stay as logged -- they were evaluated at the
        # solver's `qd`, and there is no analytic arm model here to re-evaluate them. c_arm
        # is 2.2 N*m rms against `M qdd`'s 12.1, and the ledger improves anyway.
        self.fd_clock = bool(fd_clock)
        self.filter_velocity_saturated = bool(filter_velocity_saturated)
        # Which per-episode physical parameter the contrastive context loss is structured
        # by. "mass" reads object_dynamics/mass; "arm_damping" reads the arm's PD damping
        # gain from robot_joint_params/joint_damping, which is the randomized axis in the
        # damping datasets. Kept separate from object_context so its dimensionality (and
        # every consumer of it) is unchanged.
        if context_label_key not in CONTEXT_LABEL_KEYS:
            raise ValueError(
                f"context_label_key must be one of {sorted(CONTEXT_LABEL_KEYS)}, got {context_label_key!r}"
            )
        self.context_label_key = context_label_key
        self.layout = layout or RobotObjectWMStateLayout()

        self.history_states: list[np.ndarray] = []
        self.history_actions: list[np.ndarray] = []
        self.history_torques: list[np.ndarray] = []
        self.future_actions: list[np.ndarray] = []
        self.future_torques: list[np.ndarray] = []
        self.future_qdd: list[np.ndarray] = []
        self.future_mcg_H: list[np.ndarray] = []
        self.future_mcg_c: list[np.ndarray] = []
        self.future_mcg_g: list[np.ndarray] = []
        self.future_states: list[np.ndarray] = []
        self.object_context: list[np.ndarray] = []
        self.context_label: list[np.ndarray] = []
        self.contact_t_in_window: list[int] = []
        self.sample_meta: list[tuple[str, str, int, int]] = []
        # Episode identity per window. Study 04 needs it as the *only* grouping signal
        # for a label-free InfoNCE: two windows of the same episode share a payload
        # without anybody reading the mass. `window_start` lets a sampler demand a
        # temporal gap between the two views so they do not share history frames.
        self.episode_id: list[int] = []
        self.window_start: list[int] = []
        self.episode_keys: list[tuple[str, str]] = []

        # False as soon as one accepted episode lacks robot_dynamics/qdd; the windows are
        # still built (zero-filled) so the vision recording stays loadable, but the
        # inverse-dynamics loss must refuse to run against a zero target.
        self.has_qdd = True
        # Sidecar with GT_full = logged arm M,C,G + analytic payload, written by
        # cdelan.analysis.build_mcg_targets on the states clock at full episode length. Sliced
        # here with EXACTLY the treatment robot_dynamics/qdd gets, so the two stay aligned by
        # construction. None -> the MCG windows are zeros and has_mcg stays False.
        self.mcg_target_file = mcg_target_file
        self.has_mcg = mcg_target_file is not None
        self.skipped_short_episodes = 0
        self.skipped_missing_torque_episodes = 0
        self.skipped_missing_object_dynamics_episodes = 0
        self.skipped_contact_filtered_windows = 0
        self.skipped_phase_filtered_windows = 0
        self.skipped_saturated_windows = 0
        self.episodes_with_contact = 0

        self._build_samples(
            episode_refs=episode_refs,
            filter_pre_contact=filter_pre_contact,
            object_displacement_threshold=object_displacement_threshold,
            object_velocity_threshold=object_velocity_threshold,
            contact_consecutive_steps=contact_consecutive_steps,
            contact_settle_steps=contact_settle_steps,
        )

    def _material_context(self, object_dyn_group, t_count: int) -> np.ndarray:
        if "material_properties" not in object_dyn_group:
            return np.zeros((t_count, self.layout.object_material_dim), dtype=np.float32)
        material = np.asarray(object_dyn_group["material_properties"], dtype=np.float32)[:t_count]
        material = material.reshape(material.shape[0], -1)
        if material.shape[1] < self.layout.object_material_dim:
            pad = np.zeros(
                (material.shape[0], self.layout.object_material_dim - material.shape[1]),
                dtype=np.float32,
            )
            material = np.concatenate([material, pad], axis=-1)
        return material[:, : self.layout.object_material_dim]

    def _env_origin(self, episode) -> np.ndarray:
        try:
            init_group = episode["initial_state"]["articulation"]["robot"]
            if "root_pose" in init_group:
                return np.asarray(init_group["root_pose"], dtype=np.float32)[0, : self.layout.object_pos_dim]
        except KeyError:
            pass
        return np.zeros(self.layout.object_pos_dim, dtype=np.float32)

    def _build_samples(
        self,
        episode_refs: List[EpisodeRef],
        filter_pre_contact: bool,
        object_displacement_threshold: float,
        object_velocity_threshold: float,
        contact_consecutive_steps: int,
        contact_settle_steps: int,
    ) -> None:
        by_file: dict[str, list[str]] = defaultdict(list)
        for path, name in episode_refs:
            by_file[path].append(name)

        k = self.history_len - 1
        h = self.rollout_horizon

        for hdf5_path, names in by_file.items():
            with h5_open(hdf5_path) as file:
                data_group = file[Hdf5Groups.DATA]
                for episode_name in names:
                    episode = data_group[episode_name]
                    obs = episode[Hdf5Groups.OBS]
                    if Hdf5Groups.ROBOT_TORQUES not in episode or self.torque_key not in episode[Hdf5Groups.ROBOT_TORQUES]:
                        self.skipped_missing_torque_episodes += 1
                        continue
                    if Hdf5Groups.OBJECT_DYNAMICS not in episode or Hdf5Groups.STATES not in episode:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    odg = episode[Hdf5Groups.OBJECT_DYNAMICS]
                    if "mass" not in odg or "inertia" not in odg:
                        self.skipped_missing_object_dynamics_episodes += 1
                        continue

                    object_state_group = episode[Hdf5Groups.STATES]["rigid_object"]["object"]
                    root_pose = np.asarray(object_state_group["root_pose"], dtype=np.float32)
                    root_velocity = np.asarray(object_state_group["root_velocity"], dtype=np.float32)
                    object_pos_w = root_pose[:, : self.layout.object_pos_dim]
                    object_quat = root_pose[
                        :,
                        self.layout.object_pos_dim : self.layout.object_pos_dim + self.layout.object_quat_dim,
                    ]
                    object_lin_vel = root_velocity[:, : self.layout.object_lin_vel_dim]
                    if self.recompute_object_ang_vel:
                        object_ang_vel = _quat_to_body_ang_vel(object_quat, self.dt)
                    else:
                        object_ang_vel = root_velocity[
                            :,
                            self.layout.object_lin_vel_dim : self.layout.object_lin_vel_dim + self.layout.object_ang_vel_dim,
                        ]
                    object_pos = object_pos_w - self._env_origin(episode)[None, :] if self.subtract_env_origin else object_pos_w

                    # robot_dynamics/* is logged on the states/* clock, like the object
                    # arrays and unlike obs/* -- see the align_object_clock note below and
                    # cdelan/analysis/mcg_vs_gt.py, which pairs them the same way. This is
                    # Isaac's instantaneous acceleration at the sim step, self-consistent
                    # with its own M and tau; it is NOT the decimated finite difference the
                    # Euler integrator effectively realises over one recorded interval
                    # (r = 0.36 between the two). That makes it the right target for an
                    # inverse-dynamics loss and the wrong one for the rollout.
                    if Hdf5Groups.ROBOT_DYNAMICS in episode and "qdd" in episode[Hdf5Groups.ROBOT_DYNAMICS]:
                        qdd = np.asarray(
                            episode[Hdf5Groups.ROBOT_DYNAMICS]["qdd"], dtype=np.float32
                        )[:, : self.layout.robot_dof]
                    else:
                        self.has_qdd = False
                        qdd = np.zeros((root_pose.shape[0], self.layout.robot_dof), dtype=np.float32)

                    n_arm = _MCG_ARM_DOF
                    if self.mcg_target_file is not None:
                        with h5py.File(self.mcg_target_file, "r") as fmcg:
                            if episode_name not in fmcg["data"]:
                                self.has_mcg = False
                                mcg_H = np.zeros((root_pose.shape[0], n_arm, n_arm), dtype=np.float32)
                                mcg_c = np.zeros((root_pose.shape[0], n_arm), dtype=np.float32)
                                mcg_g = np.zeros((root_pose.shape[0], n_arm), dtype=np.float32)
                            else:
                                grp = fmcg["data"][episode_name]
                                mcg_H = np.asarray(grp["H"], dtype=np.float32)
                                mcg_c = np.asarray(grp["c"], dtype=np.float32)
                                mcg_g = np.asarray(grp["g"], dtype=np.float32)
                    else:
                        mcg_H = np.zeros((root_pose.shape[0], n_arm, n_arm), dtype=np.float32)
                        mcg_c = np.zeros((root_pose.shape[0], n_arm), dtype=np.float32)
                        mcg_g = np.zeros((root_pose.shape[0], n_arm), dtype=np.float32)

                    joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:, : self.layout.robot_dof]
                    joint_vel = np.asarray(obs["joint_vel"], dtype=np.float32)[:, : self.layout.robot_dof]
                    if self.fd_clock:
                        joint_vel, qdd = _fd_clock_channels(joint_pos, self.dt)
                    actions = np.asarray(episode[Hdf5Groups.ACTIONS], dtype=np.float32)[:, : self.layout.action_dim]
                    torques = np.asarray(episode[Hdf5Groups.ROBOT_TORQUES][self.torque_key], dtype=np.float32)[
                        :, : self.layout.torque_dim
                    ]

                    # Effort-limit saturation, on the same (obs) clock as `torques`. The
                    # gripper columns are excluded: the fingers are commanded closed and sit
                    # against their own limit for the whole of phase 3, so including them
                    # would reject every window.
                    saturated = None
                    if self.filter_velocity_saturated:
                        if "velocity_saturated" not in obs:
                            raise KeyError(
                                "filter_velocity_saturated needs obs/velocity_saturated; "
                                "this dataset predates the flag."
                            )
                        saturated = np.asarray(obs["velocity_saturated"], dtype=bool)
                    if self.filter_effort_saturated:
                        tg = episode[Hdf5Groups.ROBOT_TORQUES]
                        if "computed_torque" not in tg or "applied_torque" not in tg:
                            raise KeyError(
                                "filter_effort_saturated needs robot_torques/"
                                "{computed_torque,applied_torque}; this dataset lacks them."
                            )
                        commanded = np.asarray(tg["computed_torque"], dtype=np.float32)[:, :_MCG_ARM_DOF]
                        delivered = np.asarray(tg["applied_torque"], dtype=np.float32)[:, :_MCG_ARM_DOF]
                        effort = (
                            np.abs(commanded - delivered) > _EFFORT_SATURATION_TOL
                        ).any(axis=1)
                        saturated = effort if saturated is None else (saturated | effort)

                    mass = np.asarray(odg["mass"], dtype=np.float32).reshape(-1, self.layout.object_mass_dim)
                    inertia = np.asarray(odg["inertia"], dtype=np.float32).reshape(-1, self.layout.object_inertia_dim)

                    phase = None
                    if self.phase_filter is not None:
                        if "phase" not in obs:
                            self.skipped_missing_object_dynamics_episodes += 1
                            continue
                        phase = np.asarray(obs["phase"]).reshape(-1).astype(np.int64)

                    if self.align_object_clock:
                        # The obs/* arrays lag the states/* arrays by exactly one index
                        # (obs/joint_pos[t+1] == states/joint_position[t] - default_q, to
                        # 4e-7 rad), so a raw state vector holds the robot at physical
                        # time (i-1)dt and the object at i*dt -- 0.84 cm of gripper travel
                        # apart. Drop the first obs-clock sample and the last states-clock
                        # sample to put both halves on the same clock. Torque, actions and
                        # phase live on the obs clock and move with the robot.
                        joint_pos, joint_vel = joint_pos[1:], joint_vel[1:]
                        actions, torques = actions[1:], torques[1:]
                        if phase is not None:
                            phase = phase[1:]
                        if saturated is not None:
                            saturated = saturated[1:]
                        object_pos, object_quat = object_pos[:-1], object_quat[:-1]
                        object_lin_vel, object_ang_vel = object_lin_vel[:-1], object_ang_vel[:-1]
                        qdd = qdd[:-1]
                        mcg_H, mcg_c, mcg_g = mcg_H[:-1], mcg_c[:-1], mcg_g[:-1]

                    t_count = min(
                        joint_pos.shape[0],
                        joint_vel.shape[0],
                        actions.shape[0],
                        torques.shape[0],
                        object_pos.shape[0],
                        object_quat.shape[0],
                        object_lin_vel.shape[0],
                        object_ang_vel.shape[0],
                        qdd.shape[0],
                        mcg_H.shape[0],
                        mass.shape[0],
                        inertia.shape[0],
                    )
                    if t_count <= k + h:
                        self.skipped_short_episodes += 1
                        continue

                    # Per-episode context label for the contrastive loss. Damping is
                    # constant within an episode; joint 0 is an arm joint and every arm
                    # joint is scaled together, so one column is the whole story.
                    # Always (T, D): D = 1 for the scalar axes, 2 for friction+armature. A
                    # vector label lets every randomized parameter be supervised, instead of
                    # one of them standing in for a context that is really higher-dimensional.
                    if self.context_label_key == "arm_damping":
                        label_series = np.asarray(
                            episode["robot_joint_params"]["joint_damping"], dtype=np.float32
                        )[:t_count, :1]
                    elif self.context_label_key == "friction_armature":
                        jp = episode["robot_joint_params"]
                        label_series = np.stack(
                            [
                                np.asarray(jp["joint_friction_coeff"], dtype=np.float32)[:t_count, 0],
                                np.asarray(jp["joint_armature"], dtype=np.float32)[:t_count, 0],
                            ],
                            axis=-1,
                        )
                    elif self.context_label_key == "com_offset":
                        # The CoM study's axis on its own: the body-x offset alpha, read from
                        # object_dynamics/com (the CoM pose in the body frame, so column 0 is
                        # alpha and 1:3 are pinned to zero by the task). This is the direct
                        # transplant of the mass recipe -- one scalar structuring the
                        # proportional contrastive distance. Mass is still randomized in this
                        # dataset but is deliberately NOT in the label: supervising both at
                        # once let mass, the far easier axis, absorb the whole latent.
                        label_series = np.asarray(odg["com"], dtype=np.float32)[:t_count, :1]
                    elif self.context_label_key == "mass_com":
                        # The two independently drawn axes of the CoM study: payload mass and the
                        # CoM's body-x offset alpha. object_dynamics/com is (T, 7) -- the CoM pose
                        # in the body frame -- so column 0 is alpha and columns 1:3 are pinned to
                        # zero by the task. Both are constant within an episode.
                        label_series = np.stack(
                            [
                                np.asarray(mass, dtype=np.float32)[:t_count].reshape(-1),
                                np.asarray(odg["com"], dtype=np.float32)[:t_count, 0],
                            ],
                            axis=-1,
                        )
                    elif self.context_label_key == "mass_com3":
                        # theta = (m, c_x, c_y, c_z): the four coordinates the rho-sensitivity
                        # study kept out of the full 10-D rigid-body vector (studies/
                        # rho_sensitivity: the top four eigendirections of G span exactly {m, c},
                        # then a 5x gap to the first inertia direction). c_z is IN on purpose --
                        # measured, body-z sits a median 42 deg off vertical, so m (r x g) does
                        # not vanish for it and its sensitivity is the same order as c_x/c_y.
                        #
                        # object_dynamics/com is (T, 7) -- the CoM pose in the body frame -- so
                        # columns 0:3 are c. On the masscom grid the quaternion block 3:7 is
                        # identity (isotropic inertia, no principal-axes rotation to write); on
                        # the virtual-body dataset it is NOT, but c is read the same way here.
                        label_series = np.concatenate(
                            [
                                np.asarray(mass, dtype=np.float32)[:t_count].reshape(-1, 1),
                                np.asarray(odg["com"], dtype=np.float32)[:t_count, :3],
                            ],
                            axis=-1,
                        )
                    elif self.context_label_key == "inertial_pi":
                        # The whole 10-D inertial parameter vector the payload was drawn from:
                        # [m, h (3), Ibar (6)] with h = m*c the first mass moment and Ibar the
                        # inertia about the BODY ORIGIN. Recorded once per episode by the
                        # inertia-uniform sweep and constant within it, so it is broadcast to the
                        # per-timestep series shape the rest of this loop expects.
                        #
                        # Read from episode_physics_randomization/object_pi rather than
                        # reassembled from object_dynamics: that group holds the sampler's own
                        # output, whereas object_dynamics/{mass,com,inertia} is PhysX's readback
                        # about the CoM, which is a different parameterization (and whose com
                        # quaternion is the principal-axes rotation, not identity).
                        #
                        # The components span four orders of magnitude (m ~ 1, h ~ 1e-2,
                        # Ibar ~ 1e-4); _standardized_label divides each by its in-batch std
                        # before the pairwise gap, so no axis is drowned out by the mass column.
                        pr_group = episode.get("episode_physics_randomization")
                        if pr_group is None or "object_pi" not in pr_group:
                            raise KeyError(
                                "context_label_key='inertial_pi' needs "
                                "episode_physics_randomization/object_pi, which only the "
                                "inertia-uniform sweep records; this dataset does not have it."
                            )
                        pi = np.asarray(pr_group["object_pi"], dtype=np.float32).reshape(-1)[:10]
                        label_series = np.broadcast_to(pi, (t_count, 10)).copy()
                    else:
                        label_series = np.asarray(mass, dtype=np.float32)[:t_count].reshape(-1, 1)

                    material = self._material_context(odg, t_count)
                    object_context_series = np.concatenate(
                        [mass[:t_count], inertia[:t_count], material[:t_count]], axis=-1
                    )
                    state = np.concatenate(
                        [
                            joint_pos[:t_count],
                            joint_vel[:t_count],
                            object_pos[:t_count],
                            object_quat[:t_count],
                            object_lin_vel[:t_count],
                            object_ang_vel[:t_count],
                        ],
                        axis=-1,
                    )
                    actions = actions[:t_count]
                    torques = torques[:t_count]
                    qdd = qdd[:t_count]
                    if saturated is not None:
                        saturated = saturated[:t_count]
                    mcg_H, mcg_c, mcg_g = mcg_H[:t_count], mcg_c[:t_count], mcg_g[:t_count]

                    contact_t = first_object_motion_timestep(
                        object_pos=object_pos[:t_count],
                        dt=self.dt,
                        displacement_threshold=object_displacement_threshold,
                        velocity_threshold=object_velocity_threshold,
                        consecutive_steps=contact_consecutive_steps,
                        settle_steps=contact_settle_steps,
                    )
                    if contact_t is not None:
                        self.episodes_with_contact += 1

                    episode_id = -1
                    lag = self.history_torque_lag
                    for t in range(k, t_count - h, self.window_stride):
                        if lag and t - k - lag < 0:
                            continue
                        if filter_pre_contact and contact_t is not None and (t + h) >= contact_t:
                            self.skipped_contact_filtered_windows += 1
                            continue
                        if phase is not None and not np.all(phase[t - k - lag : t + h + 1] == self.phase_filter):
                            self.skipped_phase_filtered_windows += 1
                            continue
                        if saturated is not None and bool(saturated[t - k - lag : t + h + 1].any()):
                            self.skipped_saturated_windows += 1
                            continue

                        self.history_states.append(state[t - k : t + 1])
                        self.history_actions.append(actions[t - k : t])
                        self.history_torques.append(torques[t - k - lag : t + 1 - lag])
                        self.future_actions.append(actions[t : t + h])
                        self.future_torques.append(torques[t : t + h])
                        self.future_qdd.append(qdd[t : t + h])
                        self.future_mcg_H.append(mcg_H[t : t + h])
                        self.future_mcg_c.append(mcg_c[t : t + h])
                        self.future_mcg_g.append(mcg_g[t : t + h])
                        self.future_states.append(state[t + 1 : t + h + 1])
                        self.object_context.append(object_context_series[t])
                        self.context_label.append(np.asarray(label_series[t], dtype=np.float32))
                        self.contact_t_in_window.append(self._contact_index(contact_t, t, h))
                        self.sample_meta.append((hdf5_path, episode_name, t - k, t + h))
                        if episode_id < 0:  # only episodes that survive the filters get an id
                            episode_id = len(self.episode_keys)
                            self.episode_keys.append((hdf5_path, episode_name))
                        self.episode_id.append(episode_id)
                        self.window_start.append(t - k)

        if len(self.history_states) == 0:
            raise RuntimeError("No valid robot-object rollout samples were built.")

        self.history_states = np.asarray(self.history_states, dtype=np.float32)
        self.history_actions = np.asarray(self.history_actions, dtype=np.float32)
        self.history_torques = np.asarray(self.history_torques, dtype=np.float32)
        self.future_actions = np.asarray(self.future_actions, dtype=np.float32)
        self.future_torques = np.asarray(self.future_torques, dtype=np.float32)
        self.future_qdd = np.asarray(self.future_qdd, dtype=np.float32)
        self.future_mcg_H = np.asarray(self.future_mcg_H, dtype=np.float32)
        self.future_mcg_c = np.asarray(self.future_mcg_c, dtype=np.float32)
        self.future_mcg_g = np.asarray(self.future_mcg_g, dtype=np.float32)
        self.future_states = np.asarray(self.future_states, dtype=np.float32)
        self.object_context = np.asarray(self.object_context, dtype=np.float32)
        self.context_label = np.asarray(self.context_label, dtype=np.float32)
        self.contact_t_in_window = np.asarray(self.contact_t_in_window, dtype=np.int64)
        self.episode_id = np.asarray(self.episode_id, dtype=np.int64)
        self.window_start = np.asarray(self.window_start, dtype=np.int64)
        order = np.argsort(self.episode_id, kind="stable")
        bounds = np.searchsorted(self.episode_id[order], np.arange(len(self.episode_keys) + 1))
        self.episode_window_indices = [order[bounds[i] : bounds[i + 1]] for i in range(len(self.episode_keys))]

    @staticmethod
    def _contact_index(contact_t: int | None, t: int, horizon: int) -> int:
        if contact_t is None:
            return -1
        if contact_t <= t:
            return -2
        if contact_t >= t + horizon + 1:
            return -1
        return int(contact_t - (t + 1))

    def __len__(self) -> int:
        return int(self.history_states.shape[0])

    def __getitem__(self, idx: int):
        return {
            "history_states": torch.from_numpy(self.history_states[idx]),
            "history_actions": torch.from_numpy(self.history_actions[idx]),
            "history_torques": torch.from_numpy(self.history_torques[idx]),
            "future_actions": torch.from_numpy(self.future_actions[idx]),
            "future_torques": torch.from_numpy(self.future_torques[idx]),
            "future_qdd": torch.from_numpy(self.future_qdd[idx]),
            "future_mcg_H": torch.from_numpy(self.future_mcg_H[idx]),
            "future_mcg_c": torch.from_numpy(self.future_mcg_c[idx]),
            "future_mcg_g": torch.from_numpy(self.future_mcg_g[idx]),
            "future_states": torch.from_numpy(self.future_states[idx]),
            "object_context": torch.from_numpy(self.object_context[idx]),
            "context_label": torch.from_numpy(np.atleast_1d(self.context_label[idx])),
            "contact_t_in_window": int(self.contact_t_in_window[idx]),
            "episode_id": int(self.episode_id[idx]),
        }
