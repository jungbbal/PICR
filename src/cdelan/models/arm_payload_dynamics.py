"""arm_payload: DeLaN over the ARM ONLY, with the grasped object as a rigid payload.

Motivation
----------
In phase 3 the object is already grasped and does not move relative to the
gripper (measured slip over an 8-step window: 0.15 cm, median 0.07 cm). Its 6
pose coordinates are therefore *not* independent degrees of freedom. The
`whole` / `whole_revise` models nevertheless put them into the generalized
coordinate vector

    q_whole = [robot_q(9), object_pos(3), object_rotvec(3)]   with force [tau, 0_6]

which (a) leaves the rigid-grasp constraint entirely up to the learned mass
matrix -- in practice violated by ~16x -- and (b) makes H(q) ill-conditioned
(cond ~1.3e3, the weak subspace concentrated on the object coordinates) even
though we invert it at every step.

This model instead solves the physically correct system

    H(q^r, z) ddq^r + C(q^r, dq^r, z) dq^r + g(q^r, z) = tau,      q^r in R^9

and *derives* the object pose from forward kinematics plus a grasp offset that
is measured once from the last history frame and held fixed across the rollout:

    p_obj = p_grip(q^r) + R_grip(q^r) d_off
    th_obj = th_grip(q^r) (x) th_off

Consequences that fall out for free:
  * the rigid-grasp constraint holds exactly, by construction;
  * H is 9x9 with no redundant coordinates;
  * the payload mass enters through H/C/g of the ARM, which is what the context
    latent z is supposed to modulate in the first place;
  * the object quaternion no longer comes from integrating the (contact-jitter
    contaminated) omega channel -- it comes from the joint angles.

The public 31-d state layout is unchanged, so the dataset, losses and eval
pipeline are reused as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import os
import torch
from torch import nn

from .context import ContextEncoder, ContextOutput, bound_latent, reparameterize
from .delan import BasisDeLaNCore, DeLaNCore, DelanTerms, ResidualDeLaNCore, validate_ode_solver
from .utils import (
    FrankaForwardKinematics,
    normalize_quat,
    quat_conjugate,
    quat_mul,
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    safe_solve,
)

# obs/joint_pos is stored relative to the robot's default pose; FK needs absolute
# angles. Measured over 150 episodes the difference is an exact constant
# (per-joint std 5.5e-6 rad) once the obs/states one-step offset is accounted for.
DEFAULT_FRANKA_JOINT_POS = (0.0, -0.569, 0.0, -2.809998, 0.0, 3.036996, 0.740999, 0.04, 0.04)


class QddResidual(nn.Module):
    """``Delta(q, dq, tau, z)``, added to the acceleration the physics term produces.

    WHY THIS EXISTS. Measured on lift_weldlat under the fd clock, k=24 h=16, through the
    campaign's own harness:

        gt_qdd   the RECORDED acceleration re-integrated      0.0013 cm at h=16
        gt_mcg   the LOGGED M, c, g pushed through M^-1       0.1692 cm at h=16

    Both rows are perfect information. The 130x between them is bought entirely by the ROUTE:
    the ledger `M qdd + c + g = tau` closes to 0.046% of |tau|, but ||tau|| / ||tau - c - g||
    is 7.1x here, so that residual arrives at the right-hand side as ~0.33% and M^-1 then
    amplifies it in its weak directions. So 0.169 cm is not a statement about how well M, c and
    g can be learned -- it is the floor of the M^-1 path itself, and no improvement to H, c or g
    goes below it.

    This term is the way out. It is added after the physics acceleration, so it is free of the
    M^-1 amplification and the model is no longer bounded by 0.169; the physics branch keeps
    carrying z and keeps answering to `mcg_supervision_loss`, which is what holds it near the
    ground truth instead of letting the residual quietly become the whole model.

    The output layer is zero-initialised, so at step 0 this model IS the physics model. The
    comparison against a run without it therefore starts from a common point rather than from a
    perturbed one, and any gap is training, not initialisation.
    """

    def __init__(
        self,
        robot_dof: int,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        tau_rms: float = 15.06,
    ) -> None:
        super().__init__()
        self.robot_dof = int(robot_dof)
        self.latent_dim = int(latent_dim)
        layers: list[nn.Module] = []
        width = 3 * self.robot_dof + self.latent_dim
        for _ in range(max(1, int(depth))):
            layers += [nn.Linear(width, hidden_dim), nn.SiLU()]
            width = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, self.robot_dof)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # tau is ~15 N*m rms on this recording while q and dq are O(1); dividing by the measured
        # scale keeps the three input blocks within an order of magnitude of each other, which a
        # plain MLP needs and which a LayerNorm would only buy by mixing them together.
        self.register_buffer("tau_rms", torch.tensor(float(tau_rms)), persistent=False)

    def forward(
        self,
        q_rel: torch.Tensor,
        dq: torch.Tensor,
        torque: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [q_rel, dq, torque / self.tau_rms]
        if self.latent_dim:
            if z is None:
                raise ValueError("QddResidual was built with latent_dim > 0 but got z=None.")
            parts.append(z)
        return self.out(self.net(torch.cat(parts, dim=-1)))


@dataclass(frozen=True)
class ArmPayloadWMDynamicsConfig:
    robot_dof: int = 9
    torque_dim: int = 9
    hidden_dim: int = 256
    dt: float = 0.02
    ode_solver: str = "euler"
    history_len: int = 12
    use_context_encoder: bool = True
    latent_dim: int = 2
    context_encoder_hidden_dim: int = 256
    context_encoder_depth: int = 2
    delan_use_film: bool = True
    delan_film_depth: int = 2
    delan_film_modulation_limit: float = 0.1
    # FiLM gate temperature T: limit * T * tanh(x / T); 1.0 = the original gate (see delan.FiLMLayer).
    delan_film_temperature: float = 1.0
    # Separate gate for the shift (beta) channel; None = same as gamma.
    delan_film_beta_modulation_limit: float | None = None
    delan_film_beta_temperature: float | None = None
    # Rotor inertia added to the DeLaN mass matrix, H -> H + delan_eps_diag * I. See
    # DeLaNCore.armature_eps; 0.0 reproduces every checkpoint trained before it existed.
    delan_eps_diag: float = 0.0
    # Let the DeLaN core factor W = H^-1 rather than H; the rollout then multiplies instead of
    # solving. See DeLaNCore.parameterize_inverse for why the gradient is the reason.
    delan_parameterize_inverse: bool = False
    # Learned acceleration residual on top of the physics term; see QddResidual for why the
    # M^-1 path alone is capped at 0.169 cm on this recording.
    qdd_residual: bool = False
    qdd_residual_hidden_dim: int = 256
    qdd_residual_depth: int = 2
    # Split H, c, g into a fitted arm branch plus a z-driven payload branch. See
    # ResidualDeLaNCore; arm_prior_ckpt is a state_dict for the arm branch from
    # cdelan.analysis.fit_arm_prior, and arm_prior_freeze holds it fixed afterwards.
    arm_prior: bool = False
    arm_prior_ckpt: str = ""
    arm_prior_freeze: bool = True
    mass_head_hidden_dim: int = 128
    predict_mass: bool = True
    tool_z_offset: float = 0.1034
    context_input_arm_only: bool = False
    # encoder / core options, all off by default (see ArmPayloadWMDynamics.encode_context
    # and BasisDeLaNCore).
    context_residual_features: bool = False
    context_input_standardize: bool = False
    delan_core: str = "film"
    delan_basis_count: int = 12
    context_ls_refine: bool = False
    context_ls_trust: float = 0.5
    # encoder history without q (dq + tau only). Off by default.
    context_drop_q: bool = False

    @property
    def state_dim(self) -> int:
        return 2 * self.robot_dof + 13


class ArmPayloadWMDynamics(nn.Module):
    """9-DOF arm DeLaN; the payload rides along via forward kinematics."""

    def __init__(
        self,
        *,
        robot_dof: int = 9,
        torque_dim: int = 9,
        hidden_dim: int = 256,
        dt: float = 0.02,
        ode_solver: str = "euler",
        history_len: int = 12,
        use_context_encoder: bool = True,
        latent_dim: int = 2,
        context_encoder_hidden_dim: int = 256,
        context_encoder_depth: int = 2,
        delan_use_film: bool = True,
        delan_film_depth: int = 2,
        delan_film_modulation_limit: float = 0.1,
        delan_eps_diag: float = 0.0,
        delan_parameterize_inverse: bool = False,
        qdd_residual: bool = False,
        delan_film_temperature: float = 1.0,
        delan_film_beta_modulation_limit: float | None = None,
        delan_film_beta_temperature: float | None = None,
        qdd_residual_hidden_dim: int = 256,
        qdd_residual_depth: int = 2,
        mass_head_hidden_dim: int = 128,
        predict_mass: bool = True,
        tool_z_offset: float = 0.1034,
        context_input_arm_only: bool = False,
        arm_prior: bool = False,
        arm_prior_freeze: bool = True,
        context_residual_features: bool = False,
        context_input_standardize: bool = False,
        delan_core: str = "film",
        delan_basis_count: int = 12,
        context_ls_refine: bool = False,
        context_ls_trust: float = 0.5,
        context_drop_q: bool = False,
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("ArmPayloadWMDynamics expects torque_dim == robot_dof.")
        self.robot_dof = int(robot_dof)
        self.torque_dim = int(torque_dim)
        self.state_dim = 2 * self.robot_dof + 13
        self.history_len = int(history_len)
        self.dt = float(dt)
        self.ode_solver = validate_ode_solver(ode_solver)
        # optional joint-velocity clamp inside the rollout (the PhysX articulation clamps |dq| at the joint velocity limit:
        # 2.175 rad/s j1-4, 2.61 j5-7 on the Franka). Known simulator constraint, not a learned term. Enabled by CDELAN_DQ_CLIP="v1,...,v7";
        # applied after each integration step to the first len(v) joints (the fingers are left alone).
        _clip = os.environ.get("CDELAN_DQ_CLIP", "")
        self.dq_clip = torch.tensor([float(x) for x in _clip.split(",")]) if _clip else None
        self.use_context_encoder = bool(use_context_encoder)
        self.context_input_arm_only = bool(context_input_arm_only)
        # the encoder reads [dq, tau] per history frame, no joint position. The dynamics core still
        # integrates q; only the estimator's input changes.
        self.context_drop_q = bool(context_drop_q)
        self.delan_use_film = bool(delan_use_film)
        self.latent_dim = int(latent_dim) if self.use_context_encoder else 0
        if self.delan_use_film and not self.use_context_encoder:
            raise ValueError("delan_use_film=True requires use_context_encoder=True.")

        # One DeLaN over the arm only -- 9 coordinates, no object DOFs.
        self.arm_prior = bool(arm_prior)
        self.delan_core = str(delan_core)
        if self.delan_core not in ("film", "basis"):
            raise ValueError(f"delan_core must be 'film' or 'basis'; got {delan_core!r}.")
        self.context_ls_refine = bool(context_ls_refine)
        self.ls_trust = float(context_ls_trust)
        if self.context_ls_refine and self.delan_core != "basis":
            raise ValueError("context_ls_refine needs delan_core='basis': only there is tau linear in z's weights.")
        self.context_residual_features = bool(context_residual_features) and self.use_context_encoder
        self.context_input_standardize = bool(context_input_standardize) and self.use_context_encoder
        self.delan_parameterize_inverse = bool(delan_parameterize_inverse)
        if self.arm_prior and self.delan_parameterize_inverse:
            raise ValueError(
                "arm_prior=True and delan_parameterize_inverse=True are mutually exclusive; "
                "see ResidualDeLaNCore.__init__ for why the inverse does not split."
            )
        if self.delan_core == "basis":
            if self.arm_prior or self.delan_parameterize_inverse or not self.delan_use_film:
                raise ValueError("delan_core='basis' needs delan_use_film=True (z is its only context "
                                 "path) and excludes arm_prior / delan_parameterize_inverse.")
            self.core = BasisDeLaNCore(
                robot_dof=self.robot_dof,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                n_basis=delan_basis_count,
                depth=delan_film_depth + 1,
                armature_eps=delan_eps_diag,
                identity_head=self.context_ls_refine,
            )
            if self.context_ls_refine:
                # relative ridge of the history least squares, softplus(raw); 0.1 at the start
                self.ls_ridge_raw = nn.Parameter(torch.tensor(-2.2522))
        elif self.arm_prior:
            if not self.delan_use_film:
                raise ValueError("arm_prior=True needs delan_use_film=True: the payload branch is "
                                 "the only place z enters, so without FiLM nothing carries it.")
            self.core = ResidualDeLaNCore(
                robot_dof=self.robot_dof,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                armature_eps=delan_eps_diag,
                film_depth=delan_film_depth,
                film_modulation_limit=delan_film_modulation_limit,
                film_temperature=delan_film_temperature,
                film_beta_modulation_limit=delan_film_beta_modulation_limit,
                film_beta_temperature=delan_film_beta_temperature,
            )
            if arm_prior_freeze:
                self.core.freeze_arm()
        else:
            self.core = DeLaNCore(
                robot_dof=self.robot_dof,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim if self.delan_use_film else None,
                armature_eps=delan_eps_diag,
                use_film=self.delan_use_film,
                film_depth=delan_film_depth,
                film_modulation_limit=delan_film_modulation_limit,
                parameterize_inverse=self.delan_parameterize_inverse,
                film_temperature=delan_film_temperature,
                film_beta_modulation_limit=delan_film_beta_modulation_limit,
                film_beta_temperature=delan_film_beta_temperature,
            )
        self.qdd_residual = (
            QddResidual(
                robot_dof=self.robot_dof,
                latent_dim=self.latent_dim if self.use_context_encoder else 0,
                hidden_dim=qdd_residual_hidden_dim,
                depth=qdd_residual_depth,
            )
            if qdd_residual
            else None
        )
        self.fk = FrankaForwardKinematics(robot_dof=self.robot_dof, tool_z_offset=tool_z_offset)
        self.register_buffer(
            "default_joint_pos",
            torch.tensor(DEFAULT_FRANKA_JOINT_POS[: self.robot_dof], dtype=torch.float32),
        )

        if self.use_context_encoder:
            if self.history_len <= 0:
                raise ValueError("use_context_encoder=True requires history_len > 0.")
            enc_in = self.history_len * (self._enc_state_dim + self.torque_dim)
            if self.context_residual_features:
                enc_in += (self.history_len - 1) * self.robot_dof
            if self.context_input_standardize or self.context_residual_features or self.context_ls_refine:
                # Fixed train-split statistics (fit_input_stats), persistent so a checkpoint carries
                # them. Fixed rather than batch-norm: the weight EMA copies buffers once, at build.
                width = self._enc_state_dim + self.torque_dim
                self.register_buffer("ctx_in_mean", torch.zeros(width))
                self.register_buffer("ctx_in_std", torch.ones(width))
                self.register_buffer("ctx_res_scale", torch.ones(self.robot_dof))
            self.context_encoder = ContextEncoder(
                seq_input_dim=enc_in,
                hidden_dim=context_encoder_hidden_dim,
                latent_dim=latent_dim,
                depth=context_encoder_depth,
            )
        else:
            self.context_encoder = None

        self.predict_mass = bool(predict_mass) and self.use_context_encoder
        if self.predict_mass:
            self.mass_head = nn.Sequential(
                nn.Linear(self.latent_dim, mass_head_hidden_dim),
                nn.SiLU(),
                nn.Linear(mass_head_hidden_dim, 1),
            )
        else:
            self.mass_head = None

    @property
    def context_state_dim(self) -> int:
        """Per-frame state width the context encoder sees.

        ``context_input_arm_only`` cuts the 13 object channels (pos, quat, lin/ang
        vel) so the encoder gets exactly what the arm exposes -- [q, dq] plus the
        torque -- and cannot read the payload's pose history. The dynamics core is
        arm-only either way; this closes the same gap on the *estimator* side, which
        is what makes the row comparable against context estimators built on
        proprioception alone.
        """
        return 2 * self.robot_dof if self.context_input_arm_only else self.state_dim

    @property
    def _enc_state_dim(self) -> int:
        """Per-frame state width actually fed to the encoder: dq only with context_drop_q, else context_state_dim."""
        return self.robot_dof if getattr(self, "context_drop_q", False) else self.context_state_dim

    def _enc_states(self, history_states):
        if getattr(self, "context_drop_q", False):
            return history_states[..., self.robot_dof : 2 * self.robot_dof]
        return history_states[..., : self.context_state_dim]

    def fit_input_stats(self, dataset) -> None:
        """Train-split mean/std of the encoder's per-frame input [state, tau]; the residual
        channels are divided by the per-joint torque std. No-op unless one of the two encoder
        options is on, so every earlier arm_payload run is untouched by the trainer's hook."""
        if not (self.context_input_standardize or self.context_residual_features or self.context_ls_refine):
            return
        import numpy as np

        d = self._enc_state_dim
        s = self._enc_states(np.asarray(dataset.history_states)).reshape(-1, d).astype(np.float64)
        t = np.asarray(dataset.history_torques).reshape(-1, self.torque_dim).astype(np.float64)
        mean = np.concatenate([s.mean(0), t.mean(0)])
        std = np.concatenate([s.std(0), t.std(0)]).clip(min=1e-3)
        dev = self.ctx_in_mean.device
        self.ctx_in_mean.copy_(torch.from_numpy(mean).float().to(dev))
        self.ctx_in_std.copy_(torch.from_numpy(std).float().to(dev))
        self.ctx_res_scale.copy_(torch.from_numpy(t.std(0)[: self.robot_dof].clip(min=1e-2)).float().to(dev))
        print(f"[encoder] input stats fitted on {len(s)} history frames; standardize="
              f"{self.context_input_standardize} residual_features={self.context_residual_features}; "
              f"tau std per joint {np.array2string(t.std(0), precision=2)}")

    def history_residual(
        self,
        history_states: torch.Tensor,
        history_torques: torch.Tensor,
        z: torch.Tensor | None = None,
        intervals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The model's own torque residual on the history's K-1 recorded intervals, (B, K-1, dof).

        ``r_i = tau_i - [H(q_i, z) qdd_i + c(q_i, dq_i, z) + g(q_i, z)]`` with the interval's
        realised acceleration ``qdd_i = (dq_{i+1} - dq_i) / dt`` -- the quantity the rollout's own
        integrator produces from tau_i, so r = 0 means "this z replays the history exactly".
        ``z=None`` evaluates the context-free nominal (z = 0): what the history says beyond the
        average arm + payload. History torque i is the one applied FROM frame i (the dataset cuts
        ``torques[t-k : t+1]`` next to ``state[t-k : t+1]``), so the last one belongs to the first
        future step and is not used here.
        """
        r = self.robot_dof
        b = history_states.shape[0]
        q = history_states[:, :-1, :r]
        dq0 = history_states[:, :-1, r : 2 * r]
        qdd = (history_states[:, 1:, r : 2 * r] - dq0) / self.dt
        tau = history_torques[:, :-1, :r]
        if intervals is not None:  # a subset of the K-1 intervals (the training loss samples a few)
            q, dq0, qdd, tau = q[:, intervals], dq0[:, intervals], qdd[:, intervals], tau[:, intervals]
        m = q.shape[1]
        if self.ode_solver == "trapezoid_mid":
            # where the integrator evaluates the terms: half-step configuration, and the interval's
            # midpoint velocity dq + qdd dt/2 (its own fixed point when the residual is zero)
            q = q + 0.5 * self.dt * dq0
            dq0 = dq0 + 0.5 * self.dt * qdd
        q, dq0, qdd, tau = (x.reshape(-1, r) for x in (q, dq0, qdd, tau))
        if z is None:
            zz = history_states.new_zeros((b * m, self.latent_dim))
        else:
            zz = z.unsqueeze(1).expand(-1, m, -1).reshape(-1, z.shape[-1])
        terms = self.core(self.absolute_q(q), dq0, z=zz if self.delan_use_film else None)
        tau_hat = torch.einsum("bij,bj->bi", terms.H, qdd) + terms.coriolis + terms.g
        return (tau - tau_hat).reshape(b, m, r)

    # ------------------------------------------------------------------ #
    # grasp bookkeeping
    # ------------------------------------------------------------------ #
    def absolute_q(self, q_rel: torch.Tensor) -> torch.Tensor:
        return q_rel + self.default_joint_pos

    def gripper_pose(self, q_rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_grip, R_grip = self.fk(self.absolute_q(q_rel))
        return p_grip, R_grip, rotation_matrix_to_quat(R_grip)

    def infer_grasp(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Grasp offset (object pose expressed in the gripper frame) from one frame.

        Held fixed for the whole rollout: in phase 3 the object does not move in
        the gripper frame, so this is the constraint the model is given.
        """
        r = self.robot_dof
        p_grip, R_grip, th_grip = self.gripper_pose(state[:, :r])
        p_obj = state[:, 2 * r : 2 * r + 3]
        th_obj = normalize_quat(state[:, 2 * r + 3 : 2 * r + 7])
        d_off = torch.einsum("bji,bj->bi", R_grip, p_obj - p_grip)  # R^T (p_obj - p_grip)
        th_off = quat_mul(quat_conjugate(th_grip), th_obj)
        return d_off, th_off

    def assemble_state(
        self,
        q_rel: torch.Tensor,
        dq: torch.Tensor,
        grasp: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Public 31-d state from arm coordinates + the frozen grasp offset."""
        d_off, th_off = grasp
        q_abs = self.absolute_q(q_rel)
        p_grip, R_grip = self.fk(q_abs)
        th_grip = rotation_matrix_to_quat(R_grip)

        lever = torch.einsum("bij,bj->bi", R_grip, d_off)  # gripper -> object, world frame
        p_obj = p_grip + lever
        th_obj = normalize_quat(quat_mul(th_grip, th_off))

        jac = self.fk.jacobian(q_abs)  # (B, 6, robot_dof)
        v_grip = torch.einsum("bij,bj->bi", jac[:, :3], dq)
        w_world = torch.einsum("bij,bj->bi", jac[:, 3:], dq)
        v_obj = v_grip + torch.cross(w_world, lever, dim=-1)
        # The dataset stores the object angular velocity in the BODY frame (that is
        # the convention integrate_quat and _quat_to_body_ang_vel use), so rotate
        # the world-frame rate from the Jacobian into the object frame.
        w_obj = torch.einsum("bji,bj->bi", quat_to_rotation_matrix(th_obj), w_world)
        return torch.cat([q_rel, dq, p_obj, th_obj, v_obj, w_obj], dim=-1)

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def acceleration(
        self,
        q_rel: torch.Tensor,
        dq: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DelanTerms]:
        # Gravity depends on the true configuration, so the core sees absolute angles.
        terms = self.core(self.absolute_q(q_rel), dq, z=z if self.delan_use_film else None)
        rhs = torque - terms.g - terms.coriolis
        # `terms.W` is H^-1, present only under delan_parameterize_inverse. Everything after the
        # operator is left identical on both paths so the ablation reads one change, not two --
        # on the W path the clamp should never bind (see DeLaNCore.parameterize_inverse).
        ddq = torch.einsum("bij,bj->bi", terms.W, rhs) if terms.W is not None \
            else safe_solve(terms.H, rhs)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        if self.qdd_residual is not None:
            # AFTER the guard and the clamp: those exist to keep a bad H^-1 from producing a
            # non-finite acceleration, and the residual is not part of that failure mode. Its
            # magnitude is carried on the terms so the training loop can both penalise it and
            # report what fraction of qdd the physics branch is still supplying.
            delta = self.qdd_residual(q_rel, dq, torque, z if self.use_context_encoder else None)
            terms.qdd_residual = delta
            ddq = ddq + delta
        return ddq, terms

    def inverse_dynamics_tau(
        self,
        q_rel: torch.Tensor,
        dq: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DelanTerms]:
        """The DeLaN terms at a GIVEN (q, dq) -- H, c, g, without solving for ddq.

        `acceleration` inverts H, and the inverse is what hides the terms' absolute
        scale. Not by creating a null direction -- ``ddq = H^-1(tau - c - g)`` and
        ``H ddq + c + g = tau`` are the same equation, so the family
        ``(H, c, g) -> (aH, ac, (1-a)tau + ag)`` is invisible to both -- but by changing
        what a departure from it costs. The realisable member of that family carries
        ``(1-a)(tau_tilde - tau)``, since a net can only approximate tau from (q, dq).
        Torque space charges that linearly in a; acceleration space divides it by aH and
        saturates. Measured on phase-3 arm joints, a: 1 -> 50 moves the qdd error
        1183 -> 1161 rad/s^2 and the torque error 8.0 -> 398.9 N*m. So the forward loss
        lets the model buy insensitivity to its own misfit by inflating H, which is what
        armpay_ctx did (H ~ 10x, c ~ 40x, g left fitting tau). Returning the terms
        un-inverted is what lets the loss be written in N*m instead. The caller supplies
        the measured ddq; see `inverse_dynamics_loss`.
        """
        terms = self.core(self.absolute_q(q_rel), dq, z=z if self.delan_use_film else None)
        return terms

    def step(
        self,
        state: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One step in the public 31-d state space.

        Same signature as the whole/whole_revise models so the analysis scripts
        (z ablation, z-swap videos, ...) work unchanged. The grasp offset is
        re-read from `state` each call; because `assemble_state` produced that
        state from the same offset, this round-trips exactly.
        """
        r = self.robot_dof
        grasp = self.infer_grasp(state)
        q_rel, dq, terms = self._step_arm(state[:, :r], state[:, r : 2 * r], torque, z=z)
        next_state = self.assemble_state(q_rel, dq, grasp)
        aux: dict[str, torch.Tensor] = {}
        if return_aux:
            aux.update(terms.as_aux())
            aux["grasp_offset"] = grasp[0]
        return next_state, aux

    def _step_arm(
        self,
        q_rel: torch.Tensor,
        dq: torch.Tensor,
        torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
        substeps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, DelanTerms]:
        """One recorded interval, optionally integrated in ``substeps`` inner steps.

        The recording is decimated: `sim_args` in the export reads dt 0.01 with
        decimation 2, so the simulator recomputed the acceleration once in the middle
        of every interval we see. Advancing the whole 0.02 s on a single acceleration
        holds it constant across a span where gravity's moment arm and the inertia
        both moved, and that discrepancy is what a ground-truth replay of the recorded
        accelerations still loses (1.11 cm at h = 8 -- the protocol's own error, not a
        model's). Splitting the interval recovers the part of it caused by the held
        acceleration.

        The torque is held across the substeps, because the midpoint torque is not
        recorded: only every second PD output is logged, and reconstructing it from
        the logged gains and targets lands a median 0.29 N*m off with outliers to
        191 N*m, which is not accurate enough to integrate against. So substepping
        fixes the held *acceleration* and not the held *torque*.
        """
        if self.ode_solver == "rk4":
            return self._rk4_step(q_rel, dq, torque, z=z)
        if self.ode_solver == "trapezoid_pc":
            return self._pc_step(q_rel, dq, torque, z=z)
        dt = self.dt / substeps
        for _ in range(substeps):
            if self.ode_solver == "trapezoid_mid":
                ddq, terms = self._mid_acceleration(q_rel, dq, torque, dt, z=z)
            else:
                ddq, terms = self.acceleration(q_rel, dq, torque, z=z)
            dq_next = dq + ddq * dt
            if self.dq_clip is not None:
                lim = self.dq_clip.to(dq_next.device, dq_next.dtype); n = lim.shape[0]
                dq_next = torch.cat([torch.maximum(torch.minimum(dq_next[..., :n], lim), -lim), dq_next[..., n:]], dim=-1)
            if self.ode_solver in ("trapezoid", "trapezoid_mid"):
                # Constant-acceleration kinematics over the interval, q + dq dt + 1/2 ddq dt^2.
                # With ddq defined as the 20 ms MEAN acceleration (which is what the rollout loss
                # makes it), this is consistent for q and dq at once; the semi-implicit update
                # below moves q with the END velocity and leaves a 1/2 * d(dq) * dt bias that the
                # model can only absorb by distorting ddq (measured 0.42 -> 0.18 cm
                # floor at h = 8 on the standard dataset with the finite-difference acceleration).
                q_rel = q_rel + 0.5 * (dq + dq_next) * dt
            else:
                q_rel = q_rel + dq_next * dt  # semi-implicit, as in the other models
            dq = dq_next
        return q_rel, dq, terms

    def _coriolis_at(self, terms: DelanTerms, dq: torch.Tensor) -> torch.Tensor:
        """The Christoffel term of already-evaluated ``terms`` at another velocity (no network pass)."""
        if terms.dH_dq is not None:
            return DeLaNCore._coriolis_from_dH(terms.dH_dq, dq)
        return DeLaNCore.coriolis_centrifugal(terms.L, terms.dL_dq, dq)

    def _mid_acceleration(self, q_rel, dq, torque, dt, *, z=None):
        """Interval-mean acceleration from ONE core evaluation at the half-step state .

        The recorded torque is the 20 ms MEAN and the target the 20 ms mean acceleration, so
        evaluating H, c, g at the interval's start is O(dt) off whenever the arm moves (gravity alone
        shifts ~0.6 N m over half a step at 2 rad/s). With ground-truth terms the h=8 ADD ceiling on
        the standard val is 0.249 cm start-point, 0.213 with H and g at mid-interval, 0.173 with c
        there too (true-velocity floor 0.162; mg4 0.193 -> 0.158). So: the network is evaluated at the
        half-step configuration q + dq dt/2, and the Coriolis term -- a cheap contraction of dH/dq,
        no second network pass -- is re-evaluated at the half-step velocity dq + ddq dt/2.
        """
        if self.arm_prior or self.delan_parameterize_inverse or self.qdd_residual is not None:
            raise ValueError("ode_solver='trapezoid_mid' supports the plain FiLM and basis cores only.")
        zz = z if self.delan_use_film else None
        terms = self.core(self.absolute_q(q_rel + 0.5 * dt * dq), dq, z=zz)
        rhs = torque - terms.g
        ddq = safe_solve(terms.H, rhs - terms.coriolis)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        for _ in range(2):  # fixed point of the half-step velocity; contracts fast, c is a small term
            c_mid = self._coriolis_at(terms, dq + 0.5 * dt * ddq)
            ddq = safe_solve(terms.H, rhs - c_mid)
            ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        terms.coriolis = c_mid
        return ddq, terms

    def _pc_step(self, q_rel, dq, torque, *, z=None):
        """Predictor-corrector on the WINDOW form of the dynamics .

        The recorded mean torque satisfies  M_bar (dq1 - dq0)/dt + c_bar + g_bar = tau_bar  with the terms
        averaged over the interval's two ends. The end state is unknown at t, so predict it with the
        start-point solve, evaluate M, c, g there, and solve again with the averaged terms. Causal (only the
        model's own prediction is used), two core evaluations per step, q moves with the trapezoid rule.
        """
        dt = self.dt
        ddq0, t0 = self.acceleration(q_rel, dq, torque, z=z)
        dq_p = dq + ddq0 * dt
        q_p = q_rel + 0.5 * (dq + dq_p) * dt
        t1 = self.core(self.absolute_q(q_p), dq_p, z=z if self.delan_use_film else None)
        H = 0.5 * (t0.H + t1.H)
        rhs = torque - 0.5 * (t0.g + t0.coriolis + t1.g + t1.coriolis)
        ddq = safe_solve(H, rhs)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
        dq_next = dq + ddq * dt
        return q_rel + 0.5 * (dq + dq_next) * dt, dq_next, t0

    def _rk4_step(self, q_rel, dq, torque, *, z=None):
        dt = self.dt
        k1_dq, terms = self.acceleration(q_rel, dq, torque, z=z)
        k2_dq, _ = self.acceleration(q_rel + 0.5 * dt * dq, dq + 0.5 * dt * k1_dq, torque, z=z)
        k3_dq, _ = self.acceleration(q_rel + 0.5 * dt * (dq + 0.5 * dt * k1_dq), dq + 0.5 * dt * k2_dq, torque, z=z)
        k4_dq, _ = self.acceleration(q_rel + dt * (dq + 0.5 * dt * k2_dq), dq + dt * k3_dq, torque, z=z)
        next_dq = dq + (dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq)
        next_q = q_rel + (dt / 6.0) * (dq + 2.0 * (dq + 0.5 * dt * k1_dq) + 2.0 * (dq + 0.5 * dt * k2_dq) + next_dq)
        return next_q, next_dq, terms

    # ------------------------------------------------------------------ #
    def forward(
        self,
        history_states: torch.Tensor,
        future_torques: torch.Tensor,
        object_context: torch.Tensor | None = None,
        history_torques: torch.Tensor | None = None,
        deterministic_context: bool | None = None,
        *,
        return_aux: bool = False,
        return_context: bool = False,
        substeps: int = 1,
    ):
        del object_context
        self._validate_rollout_inputs(history_states, future_torques)
        if history_torques is None:
            history_torques = history_states.new_zeros(
                history_states.shape[0], history_states.shape[1], self.torque_dim
            )
        context = self.encode_context(
            history_states, history_torques, deterministic_context=deterministic_context
        )
        z = context.z

        state = history_states[:, -1]
        grasp = self.infer_grasp(state)
        r = self.robot_dof
        q_rel, dq = state[:, :r], state[:, r : 2 * r]

        predictions: list[torch.Tensor] = []
        aux_steps: list[dict[str, torch.Tensor]] = []
        for step in range(future_torques.shape[1]):
            q_rel, dq, terms = self._step_arm(
                q_rel, dq, future_torques[:, step], z=z, substeps=substeps
            )
            predictions.append(self.assemble_state(q_rel, dq, grasp))
            if return_aux:
                aux = terms.as_aux()
                aux.update(context.as_aux())
                aux["grasp_offset"] = grasp[0]
                aux_steps.append(aux)

        future_states = torch.stack(predictions, dim=1)
        if return_aux and return_context:
            return future_states, aux_steps, context
        if return_aux:
            return future_states, aux_steps
        if return_context:
            return future_states, context
        return future_states

    def encode_context(
        self,
        history_states: torch.Tensor,
        history_torques: torch.Tensor | None,
        *,
        deterministic_context: bool | None = None,
    ) -> ContextOutput:
        if not self.use_context_encoder:
            return ContextOutput(mu=None, logvar=None, z=None)
        if history_torques is None:
            raise ValueError("use_context_encoder=True requires history_torques.")
        if history_states.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len {self.history_len}, got {history_states.shape[1]}.")
        context_states = self._enc_states(history_states)
        frames = torch.cat([context_states, history_torques], dim=-1)
        if self.context_input_standardize:
            frames = (frames - self.ctx_in_mean) / self.ctx_in_std
        flat = frames.flatten(start_dim=1)
        if self.context_residual_features:
            # What the history says beyond the nominal (z = 0) model, in units of each joint's
            # torque std. Raw, the encoder has to find a +-1 N m per-episode signal under a 30 N m
            # torque that moves with the pose; here it is handed over directly. No gradient: the
            # features describe the core, they are not a second objective for it. z stays the only
            # path from the history into H, c, g.
            with torch.no_grad():
                # chunked: analysis scripts encode thousands of windows at once, and each window
                # costs K-1 core evaluations here (a training batch is one chunk)
                res = torch.cat([
                    self.history_residual(hs_c, ht_c)
                    for hs_c, ht_c in zip(history_states.split(256), history_torques.split(256))
                ]) / self.ctx_res_scale
            flat = torch.cat([flat, res.clamp(-10.0, 10.0).flatten(start_dim=1)], dim=-1)
        mu, logvar = self.context_encoder(flat)
        if self.context_ls_refine:
            mu = self._ls_refine(history_states, history_torques, mu)
        if deterministic_context is None:
            deterministic_context = not self.training
        z = mu if deterministic_context else reparameterize(mu, logvar)
        context = ContextOutput(mu=mu, logvar=logvar, z=z)
        if self.predict_mass:
            context = replace(context, mass_pred=self.mass_head(mu))
        return context

    def _ls_refine(self, history_states: torch.Tensor, history_torques: torch.Tensor,
                   mu: torch.Tensor) -> torch.Tensor:
        """One ridge least-squares step on the history, in the basis core's weight space.

        With ``H = sum_k w_k H_k`` and ``V = sum_k w_k V_k`` the inverse dynamics is LINEAR in the
        weights: ``tau_i = Y_i w`` on every history interval, ``Y_i`` the model's own regressor
        (column k = body k's H_k qdd + c_k + g_k). So the encoder's weights w0 = softplus(mu + b) are
        a prior and the history is solved for the correction in closed form,

            w = w0 + (Y'Y + rho I)^-1 Y' (tau - Y w0),      rows scaled by each joint's torque std,

        with ``rho`` = a learned factor times the mean diagonal of Y'Y (scale-free). Directions the
        window identifies (at rest: the gravity torque at this pose) are corrected, the rest keep the
        prior. Returned as the z whose weights are w, so z is still the only context path and every
        consumer of ``encode_context`` sees the refined latent. Y is detached: the solve is an
        estimator built on the core, not a second loss for it; the prior and rho take the gradient.
        """
        r = self.robot_dof
        b, k = history_states.shape[:2]
        with torch.no_grad():
            dq0 = history_states[:, :-1, r : 2 * r]
            q = history_states[:, :-1, :r]
            qdd = (history_states[:, 1:, r : 2 * r] - dq0) / self.dt
            if self.ode_solver == "trapezoid_mid":
                q = q + 0.5 * self.dt * dq0
                dq0 = dq0 + 0.5 * self.dt * qdd
            q, dq0, qdd = q.reshape(-1, r), dq0.reshape(-1, r), qdd.reshape(-1, r)
            y = self.core.regressor(self.absolute_q(q), dq0, qdd)                      # (B*(K-1), r, nb)
            tau = history_torques[:, :-1, :r].reshape(-1, r) - (self.core.eps + self.core.armature_eps) * qdd
            scale = self.ctx_res_scale.reshape(1, r, 1)
            y = (y / scale).reshape(b, (k - 1) * r, -1)
            tau = (tau / scale[..., 0]).reshape(b, (k - 1) * r)
            gram = y.transpose(1, 2) @ y                                               # (B, nb, nb)
            diag_mean = torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1)                # (B,)
        w0 = self.core.weights(mu)
        rhs = torch.einsum("bnk,bn->bk", y, tau - torch.einsum("bnk,bk->bn", y, w0))
        rho = torch.nn.functional.softplus(self.ls_ridge_raw) * diag_mean + 1e-6
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
        delta = torch.linalg.solve(gram + rho.reshape(-1, 1, 1) * eye, rhs)
        # Trust region: each weight moves by at most `ls_trust` of its prior value (smoothly). The
        # prior carries the coarse context, the solve the per-episode refinement; and an untrained
        # basis set cannot throw the weights anywhere (unbounded, the first epochs start at a
        # rollout loss ~1e3).
        lim = self.ls_trust * w0
        delta = lim * torch.tanh(delta / lim)
        w = (w0 + delta).clamp_min(1.0e-4)
        return bound_latent(self.core.weights_to_z(w))

    def _validate_rollout_inputs(self, history_states: torch.Tensor, future_torques: torch.Tensor) -> None:
        if history_states.ndim != 3 or history_states.shape[-1] != self.state_dim:
            raise ValueError(f"history_states must be (batch, history_len, {self.state_dim}).")
        if future_torques.ndim != 3 or future_torques.shape[-1] != self.torque_dim:
            raise ValueError(f"future_torques must be (batch, horizon, {self.torque_dim}).")


def build_arm_payload_wm_dynamics(
    cfg: ArmPayloadWMDynamicsConfig = ArmPayloadWMDynamicsConfig(),
) -> ArmPayloadWMDynamics:
    return ArmPayloadWMDynamics(
        robot_dof=cfg.robot_dof,
        torque_dim=cfg.torque_dim,
        hidden_dim=cfg.hidden_dim,
        dt=cfg.dt,
        ode_solver=cfg.ode_solver,
        history_len=cfg.history_len,
        use_context_encoder=cfg.use_context_encoder,
        latent_dim=cfg.latent_dim,
        context_encoder_hidden_dim=cfg.context_encoder_hidden_dim,
        context_encoder_depth=cfg.context_encoder_depth,
        delan_use_film=cfg.delan_use_film,
        delan_film_depth=cfg.delan_film_depth,
        delan_film_modulation_limit=cfg.delan_film_modulation_limit,
        delan_film_temperature=getattr(cfg, "delan_film_temperature", 1.0),
        delan_film_beta_modulation_limit=getattr(cfg, "delan_film_beta_modulation_limit", None),
        delan_film_beta_temperature=getattr(cfg, "delan_film_beta_temperature", None),
        delan_parameterize_inverse=cfg.delan_parameterize_inverse,
        qdd_residual=cfg.qdd_residual,
        qdd_residual_hidden_dim=cfg.qdd_residual_hidden_dim,
        qdd_residual_depth=cfg.qdd_residual_depth,
        delan_eps_diag=cfg.delan_eps_diag,
        mass_head_hidden_dim=cfg.mass_head_hidden_dim,
        predict_mass=cfg.predict_mass,
        tool_z_offset=cfg.tool_z_offset,
        context_input_arm_only=cfg.context_input_arm_only,
        arm_prior=cfg.arm_prior,
        arm_prior_freeze=cfg.arm_prior_freeze,
        context_residual_features=cfg.context_residual_features,
        context_input_standardize=cfg.context_input_standardize,
        delan_core=cfg.delan_core,
        delan_basis_count=cfg.delan_basis_count,
        context_ls_refine=cfg.context_ls_refine,
        context_ls_trust=cfg.context_ls_trust,
        context_drop_q=getattr(cfg, "context_drop_q", False),
    )
