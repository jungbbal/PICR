from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the training environment.
    yaml = None

from cdelan import paths
from cdelan.data import object_mlp_dataset
from cdelan.data.dataset import filter_refs_by_mass
from cdelan.data.episode_view_sampler import EpisodeTwoViewBatchSampler
from cdelan.training.context_losses import (
    context_contrastive_loss,
    context_infonce_loss,
    context_kl_loss,
    inverse_dynamics_loss,
    mcg_supervision_loss,
    weighted_rollout_mse,
)
from cdelan.models.arm_payload_dynamics import (
    ArmPayloadWMDynamicsConfig,
    build_arm_payload_wm_dynamics,
)
from cdelan.training.checkpoint import checkpoint_and_log_epoch, start_training_run
from cdelan.training.losses import MetricAverager
from cdelan.training.helper import set_seed

MODEL_NAME = "WMDynamics"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "train_config.yaml"
NONE_STRINGS = {"", "none", "null", "nil"}
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
FIELD_CHOICES = {
    "model_type": ("arm_payload",),
    "id_qdd_source": ("logged", "fd"),
    "action_type": ("policy", "torque", "Torque"),
    "torque_key": ("applied_torque", "computed_torque", "measured_joint_force", "applied_torque_mean", "measured_joint_force_mean"),
    "ode_solver": ("euler", "trapezoid", "rk4", "trapezoid_pc", "trapezoid_mid"),
    "lr_schedule": ("none", "cosine"),
    "best_metric": ("loss", "rollout_loss"),
    "rollout_target": ("logged", "integrated_velocity"),
    # eval_split is deliberately unconstrained: with an external_split_json any key in that
    # file (e.g. "ood_test") is a legal split, and a fixed choice list would reject it.
    "wandb_mode": ("online", "offline", "disabled"),
}
FIELD_ALIASES = {
    "object_hidden_dim": ("object_mlp_hidden_dim",),
    "object_depth": ("object_mlp_depth",),
}


@dataclass
class TrainConfig:
    dataset_dir: str
    dataset_file: str | None
    dataset_id: str | None
    model_type: str
    history_len: int
    rollout_horizon: int
    robot_dof: int
    action_dim: int
    torque_dim: int
    action_type: str
    torque_key: str
    hidden_dim: int
    object_hidden_dim: int
    object_depth: int
    tool_z_offset: float
    dt: float
    ode_solver: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    q_weight: float
    dq_weight: float
    object_pos_weight: float
    object_quat_weight: float
    object_lin_vel_weight: float
    object_ang_vel_weight: float
    use_context_encoder: bool
    context_input_arm_only: bool
    latent_dim: int
    context_encoder_hidden_dim: int
    context_encoder_depth: int
    delan_use_film: bool
    delan_film_depth: int
    delan_film_modulation_limit: float
    delan_eps_diag: float
    lambda_context_kl: float
    lambda_mass_aux: float
    lambda_context_contrastive: float
    contrastive_temperature: float
    contrastive_mass_delta: float
    contrastive_mode: str
    mass_head_hidden_dim: int
    predict_mass: bool
    train_split: float
    seed: int
    num_workers: int
    output_dir: str
    run_name: str | None
    wandb_project_name: str
    wandb_entity: str
    wandb_name: str | None
    wandb_mode: str
    wandb_artifacts: bool
    log_every_batches: int
    train_batch_fraction: float
    train_subsample_fraction: float
    val_subsample_fraction: float
    filter_pre_contact: bool
    object_displacement_threshold: float
    object_velocity_threshold: float
    contact_consecutive_steps: int
    contact_settle_steps: int
    subtract_env_origin: bool
    phase_filter: int | None
    # Drop rollout windows containing an effort-saturated frame. In those frames the logged
    # tau does not close tau = H qdd + c + g (28 N*m residual against 0.3-1.1 elsewhere, and
    # it is not the payload), and fitting them is what drives the learned wrist inertia
    # ~100x above the true value. See RobotObjectWMRolloutDataset.filter_effort_saturated.
    filter_effort_saturated: bool
    # Drop rollout windows containing a frame where any arm joint is on its VELOCITY wall
    # (obs/velocity_saturated). Same argument, different constraint: there the solver's
    # limit force closes the equation of motion and no logged field carries it. See
    # RobotObjectWMRolloutDataset.filter_velocity_saturated for the measured cost.
    filter_velocity_saturated: bool
    # Derive qd and qdd from the logged POSITION instead of reading the logged velocity and
    # acceleration channels, so the three describe one trajectory. The recording's q and qd
    # disagree by 21% rms even at 0.01 s and no one-step integrator closes it; on the derived
    # clock a ground-truth replay hits the FK/grasp floor exactly (ADD 1.036 -> 0.196 cm at
    # 0.16 s, all anchors) and the ledger improves rather than degrading (0.901 -> 0.487 N*m
    # rms, wall-clean). See RobotObjectWMRolloutDataset.fd_clock.
    fd_clock: bool
    context_label_key: str
    recompute_object_ang_vel: bool
    align_object_clock: bool
    train_mass_min: float
    train_mass_max: float
    loss_normalize_channels: bool
    loss_arm_dof: int | None
    external_split_json: str
    external_split_val_key: str
    eval_after_train: bool
    eval_fail_on_error: bool
    eval_output_dir: str
    eval_split: str
    eval_batch_size: int | None
    eval_max_batches: int | None
    eval_max_episodes: int | None
    eval_num_workers: int
    eval_episode_plot: bool
    eval_episode_index: int
    eval_episode_name: str | None
    eval_start_t: int
    eval_rollout_steps: int
    eval_target: str
    eval_prediction_metrics: bool
    eval_pred_horizon: int
    eval_prediction_max_episodes: int
    eval_video: bool
    eval_fps: int
    eval_max_frames: int
    # How a multi-dimensional context_label is turned into contrastive losses.
    # "none"   = one loss against the merged gap (the easier axis takes the latent)
    # "shared" = one loss per factor, all on the whole latent
    # "blocks" = one loss per factor, each on its own disjoint slice (needs 2 dims/factor)
    contrastive_per_factor: str = "none"
    # How a multi-dimensional context_label is put on a common scale before the merged gap.
    # "std"   = divide each coordinate by its in-batch std (a property of the sampling
    #           distribution: a narrowly-visited axis gets silently amplified).
    # "range" = map each onto [0, 1] by its admissible range, contrastive_label_range. The
    #           resulting d_Theta is a fixed metric on the parameter space, identical in
    #           every batch and every run. Ignored unless contrastive_mode='proportional'.
    contrastive_label_norm: str = "std"
    # (D, 2) admissible bounds [theta_min, theta_max] per label coordinate, required by
    # contrastive_label_norm='range'. NOT read from the data: the point is that the metric
    # is the physical parameter space's, not the recording's.
    contrastive_label_range: list | None = None
    # Label-free InfoNCE: positives are two windows of the SAME episode, so the payload
    # mass never enters the loss. > 0 switches both loaders to the two-view sampler.
    lambda_context_infonce: float = 0.0
    infonce_temperature: float = 0.1
    # what counts as a positive for the InfoNCE term. "episode" = the original label-free
    # same-episode form; "mass_com_soft" = SupCon with soft positives from the physical label (context_label_key
    # mass_com3): same mass level, weight exp(-|dCoM|^2 / 2 sigma^2); "mass" = hard same-level positives.
    infonce_positives: str = "episode"
    infonce_mass_delta: float = 0.15
    infonce_com_sigma: float = 0.01
    infonce_mass_sigma: float = 0.15
    # DeLaN's own objective, ||H qdd + c + g - tau||^2, evaluated teacher-forced at the
    # logged (q, dq, qdd, tau). The rollout loss cannot see the terms' absolute scale --
    # it only constrains H^-1(tau - c - g) -- so without this the decomposition is free
    # to sit anywhere on the (aH, ac, (1-a)tau + ag) family. See inverse_dynamics_loss.
    lambda_inverse_dynamics: float = 0.0
    # Weight on the state-prediction rollout. 0.0 trains the physics from the torque
    # residual alone; the rollout is then still evaluated (under no_grad) so val_rollout
    # stays comparable across runs.
    lambda_rollout: float = 1.0
    # What the rollout loss compares the predicted q and object position against.
    # "logged" = the recorded states (every run before .
    # "integrated_velocity" = the TRUE joint velocities pushed through the model's own integrator
    # from the anchor state (q~), and the object position FK(q~) with the anchor's rigid grasp. A
    # model with perfect velocities then has exactly zero loss: the part no acceleration can fix
    # (within-step velocity curvature, cube slip in the hand; 0.16 cm ADD at h = 8 on the standard
    # val) leaves the target instead of pulling on ddq. The velocity term is unchanged.
    rollout_target: str = "logged"
    # "cosine": lr decays from cfg.lr to 0.1 x cfg.lr over cfg.epochs (stepped per epoch); "none" = constant.
    lr_schedule: str = "none"
    # Exponential moving average of the weights (0 = off). When on, validation and every saved checkpoint
    # use the averaged weights; the raw weights keep training.
    ema_decay: float = 0.0
    # Which validation number picks best.pt: "loss" (the total, dominated by InfoNCE in the IC runs) or
    # "rollout_loss" (the prediction term; the logged-target one when rollout_target != logged).
    best_metric: str = "loss"
    # Divide each joint's inverse-dynamics residual by the train split's rms tau for that
    # joint, so lambda_inverse_dynamics is dimensionless and joints 1 and 3 do not swamp it.
    id_loss_normalize: bool = True
    # Joints the inverse-dynamics loss scores. 7 = the arm; the two fingers balance grip
    # contact forces this model has no term for.
    id_loss_arm_dof: int = 7
    # "logged" = robot_dynamics/qdd, Isaac's instantaneous acceleration, self-consistent
    # with the rigid-body identity and therefore the physically correct target.
    # "fd" = (dq[t+1] - dq[t]) / dt, what the Euler integrator actually realises over one
    # recorded interval. The two agree only at r = 0.36 because the recording is decimated,
    # so which one is used decides whether this loss and the rollout pull the same way.
    id_qdd_source: str = "logged"
    # --- MCG supervision -------------------------------------------------------------------
    # Term-by-term supervision of the DeLaN decomposition against GT_full (logged arm M,C,G plus
    # the analytic payload), from the sidecar built by cdelan.analysis.build_mcg_targets. This is
    # the only objective in the study that pins the SPLIT rather than the sum -- see
    # mcg_supervision_loss. It uses privileged information (the true m, c, I enter through the
    # target), so it is a training-time signal only: at eval the model still has to read z off
    # the history like the other arms.
    lambda_mcg: float = 0.0
    mcg_target_file: str = ""
    mcg_loss_normalize: bool = True
    mcg_loss_arm_dof: int = 7
    mcg_weight_H: float = 1.0
    mcg_weight_c: float = 1.0
    mcg_weight_g: float = 1.0
    # --- residual core ---------------------------------------------------------------------
    # H = H_arm(q) + dH(q, z), and the same split for c and g. The arm branch is fitted once
    # against the LOGGED robot_dynamics/* -- a clean supervised regression, unlike the payload,
    # which is the part no measurement isolates -- then frozen, so z drives only the payload.
    # Unlike lambda_mcg this is not privileged: a robot knows its own arm, not its payload.
    arm_prior: bool = False
    arm_prior_ckpt: str = ""
    arm_prior_freeze: bool = True
    # --- inverse-inertia parameterization ---------------------------------------------------
    # The DeLaN factor builds W = H^-1 instead of H, so the rollout multiplies where it used to
    # solve. The reason is the gradient, not the forward numerics: solving puts the inverse in
    # the backward pass TWICE, so a soft direction of H sees its gradient scaled by cond(H)^2
    # and the optimiser buys quiet by flattening the spectrum -- which is visible in the fitted
    # H, cond 31 against the logged GT's 1060. Factoring W drops that to cond(H) and leaves H a
    # derived quantity. Mutually exclusive with arm_prior: the branches add in H, not in H^-1.
    delan_parameterize_inverse: bool = False
    # --- learned acceleration residual --------------------------------------------------------
    # qdd = W(tau - c - g) + Delta(q, dq, tau, z). Measured on this recording, a rollout that
    # forms M^-1 is capped at 0.169 cm ADD at h=16 even with the LOGGED M, c, g -- the ledger
    # closes to 0.046% of |tau| but ||tau||/||tau-c-g|| is 7.1x, so M^-1 amplifies what is left.
    # The residual is how a structured model gets out from under it while
    # the physics branch keeps carrying z and answering to lambda_mcg. Zero-initialised output:
    # at step 0 this is exactly the model without it. See models.QddResidual.
    qdd_residual: bool = False
    qdd_residual_hidden_dim: int = 256
    qdd_residual_depth: int = 2
    # Minimum start-index distance between an episode's two views. >= history_len means
    # they share no history frames, so the pair cannot be solved off the overlap.
    infonce_min_view_gap: int = 12
    # Anchor every N-th timestep when building rollout windows (1 = every one). For the 0.5 ms
    # substep datasets, which have 40x the rows of the 20 ms ones.
    window_stride: int = 1
    # history torque window torques[t-k-lag : t+1-lag] (0 = original, tau_t included;
    # 1 = tau_{t-12..t-1}, encoder sees past torques only). See RobotObjectWMRolloutDataset.
    history_torque_lag: int = 0
    # --- let z carry the per-episode ARM (link masses are randomised +-10%) ------
    # Encoder input: also the model's own torque residual on the history intervals at the
    # nominal z = 0 (ArmPayloadWMDynamics.history_residual), and train-split standardization of
    # the raw [state, tau] frames. History -> z stays the only context path.
    context_residual_features: bool = False
    context_input_standardize: bool = False
    # ||tau_i - ID(q_i, dq_i, qdd_i; z)||^2 over the HISTORY intervals with the encoder's own z,
    # per-joint normalised like the inverse-dynamics loss: z must replay the window it was read
    # from. At rest this pins g(q; z) to the measured torque.
    lambda_history_id: float = 0.0
    history_id_intervals: int = 4
    # Teacher-forced acceleration label MSE between the model's qdd at the LOGGED (q, dq, tau)
    # of every rollout frame (history last frame + the first h-1 future frames, the frames the ID loss
    # scores) and the finite-difference label (dq_{j+1} - dq_j) / dt, i.e. what the trapezoid update
    # realises over the interval. Each joint is divided by the train split's std of that label
    # (qdd_label_normalize). This is the 1-step counterpart of the rollout loss: same label, identity
    # weighting over the h frames, no feedback of the model's own state. Pair with lambda_rollout 0.
    lambda_qdd_label: float = 0.0
    qdd_label_normalize: bool = True
    # FiLM gate temperature scale = 1 + limit * T * tanh(gamma / T). 1.0 = original.
    delan_film_temperature: float = 1.0
    # Separate gate for the FiLM shift (beta) channel: None = same limit / temperature as gamma.
    delan_film_beta_modulation_limit: float | None = None
    delan_film_beta_temperature: float | None = None
    # Rollout-with-qdd-loss user: "loss = acceleration from the velocity difference", inputs fed
    # back): the model rolls out exactly as for the rollout loss (its own integrated state is the next input),
    # but the loss is on the per-step velocity-difference acceleration of the PREDICTED trajectory,
    # (dq^_{h+1} - dq^_h) / dt with dq^_0 = the logged anchor velocity, against the same finite-difference
    # label (dq_{h+1} - dq_h) / dt, each joint / the train split's label std (qdd_label_normalize).
    # Pair with lambda_rollout 0; val rollout_loss stays the metric.
    lambda_rollout_qdd: float = 0.0
    # "film" = one DeLaN core with FiLM-modulated hidden units.
    # "basis" = BasisDeLaNCore: H = sum_k w_k(z) L_k L_k', V = sum_k w_k(z) V_k, w >= 0.
    delan_core: str = "film"
    delan_basis_count: int = 12
    # Basis core only: refine the encoder's weights by one ridge least-squares step on the model's
    # own regressor over the history intervals (ArmPayloadWMDynamics._ls_refine). Needs
    # latent_dim == delan_basis_count.
    context_ls_refine: bool = False
    # largest relative change of a weight the least-squares step may make (smooth tanh limit)
    context_ls_trust: float = 0.5
    # Split latent InfoNCE acts on the first N dims of z only (0 = all of z). The block
    # keeps the label-free, mass-organised latent of the small-z model; the remaining dims are free
    # to carry what the rollout needs (the randomised arm) without the contrastive pressure.
    infonce_latent_dims: int = 0
    # the context encoder reads [dq, tau] per history frame, no q. Dynamics unchanged.
    context_drop_q: bool = False


def _load_config_file(path: str) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required for TrainConfig YAML. Install it with `pip install pyyaml`.")

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Train config YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(values).__name__}.")

    valid_keys = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(values) - valid_keys)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown train config key(s) in {config_path}: {joined}")
    for key in ("dataset_dir", "dataset_file", "output_dir", "external_split_json"):
        value = values.get(key)
        if not isinstance(value, str) or not value:
            continue
        value = paths.expand(value)
        if not os.path.isabs(value):
            value = str((config_path.parent / value).resolve())
        values[key] = value
    return values


def _require_config_keys(values: dict[str, Any], path: str | Path) -> None:
    required = {field.name for field in fields(TrainConfig)}
    missing = sorted(required - set(values))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing train config key(s) in {path}: {joined}")


def _defaults_from_config(argv: list[str] | None) -> tuple[dict[str, Any], str | None]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    known, _ = config_parser.parse_known_args(argv)

    default_values = _load_config_file(str(DEFAULT_CONFIG_PATH))
    _require_config_keys(default_values, DEFAULT_CONFIG_PATH)
    defaults = dict(default_values)
    config_path = str(Path(known.config).expanduser()) if known.config else str(DEFAULT_CONFIG_PATH)
    if Path(config_path).resolve() != DEFAULT_CONFIG_PATH.resolve():
        defaults.update(_load_config_file(config_path))
        _require_config_keys(defaults, config_path)
    return defaults, config_path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_STRINGS:
        return True
    if normalized in FALSE_STRINGS:
        return False
    expected = ", ".join(sorted(TRUE_STRINGS | FALSE_STRINGS))
    raise argparse.ArgumentTypeError(f"expected a boolean value ({expected}), got {value!r}")


def _resolve_annotation(expr: Any, globalns: dict[str, Any]) -> Any:
    """Resolve a (possibly string) annotation, supporting PEP 604 ``A | B``
    unions on Python < 3.10 where ``type.__or__`` is unavailable."""
    if not isinstance(expr, str):
        return expr
    expr = expr.strip()
    if "|" in expr:
        parts = [_resolve_annotation(part, globalns) for part in expr.split("|")]
        return Union[tuple(parts)]
    if expr in ("None", "NoneType"):
        return type(None)
    return eval(expr, globalns)  # noqa: S307 - trusted dataclass annotations


def _safe_get_type_hints(cls: type) -> dict[str, Any]:
    """``typing.get_type_hints`` that tolerates PEP 604 unions on Python < 3.10."""
    try:
        return get_type_hints(cls)
    except TypeError:
        globalns = dict(getattr(sys.modules.get(cls.__module__), "__dict__", {}))
        return {
            name: _resolve_annotation(annotation, globalns)
            for name, annotation in getattr(cls, "__annotations__", {}).items()
        }


def _allows_none(annotation: Any) -> bool:
    return type(None) in get_args(annotation)


def _without_none(annotation: Any) -> Any:
    all_args = get_args(annotation)
    if not all_args:
        return annotation
    args = tuple(arg for arg in all_args if arg is not type(None))
    if not args:
        return str
    if len(args) == 1:
        return args[0]
    return annotation


def _parse_json_list(value: str) -> list:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError(f"expected a JSON list, got {value!r}")
    return parsed


def _value_parser(name: str, annotation: Any):
    allow_none = _allows_none(annotation)
    value_type = _without_none(annotation) if allow_none else annotation
    origin = get_origin(value_type)
    if origin is not None:
        value_type = origin

    if value_type is bool:
        parser = _parse_bool
    elif value_type is int:
        parser = int
    elif value_type is float:
        parser = float
    elif value_type is str:
        parser = str
    elif value_type is list:
        # A nested list (contrastive_label_range) is natural in YAML but has no argparse
        # scalar form; on the command line it is JSON: '[[0.1,2.0],[-0.019,0.019]]'.
        parser = _parse_json_list
    else:
        raise TypeError(f"Unsupported TrainConfig field type for {name}: {annotation!r}")

    if not allow_none:
        return parser

    def parse_optional(value: str) -> Any:
        if str(value).strip().lower() in NONE_STRINGS:
            return None
        return parser(value)

    return parse_optional


def _option_strings(name: str, *, negative: bool = False) -> list[str]:
    option_names = [name, *FIELD_ALIASES.get(name, ())]
    flags: list[str] = []
    for option_name in option_names:
        cli_name = f"no_{option_name}" if negative else option_name
        flags.append(f"--{cli_name}")
        hyphen_name = cli_name.replace("_", "-")
        if hyphen_name != cli_name:
            flags.append(f"--{hyphen_name}")
    return flags


def _build_parser(defaults: dict[str, Any], config_path: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DeLaN + object-MLP WMDynamics model.")
    parser.add_argument("--config", type=str, default=config_path, help="Optional YAML file with TrainConfig values.")
    overrides = parser.add_argument_group("TrainConfig overrides")
    type_hints = _safe_get_type_hints(TrainConfig)
    for config_field in fields(TrainConfig):
        name = config_field.name
        annotation = type_hints[name]
        default = defaults[name]
        kwargs = {
            "dest": name,
            "default": default,
            "help": f"Override TrainConfig.{name} (default: %(default)s).",
        }
        if name in FIELD_CHOICES:
            kwargs["choices"] = FIELD_CHOICES[name]

        if _without_none(annotation) is bool:
            overrides.add_argument(
                *_option_strings(name),
                nargs="?",
                const=True,
                type=_parse_bool,
                metavar="{true,false}",
                **kwargs,
            )
            overrides.add_argument(
                *_option_strings(name, negative=True),
                dest=name,
                action="store_false",
                help=f"Set TrainConfig.{name}=False.",
            )
        else:
            overrides.add_argument(*_option_strings(name), type=_value_parser(name, annotation), **kwargs)
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    defaults, config_path = _defaults_from_config(argv)
    parser = _build_parser(defaults, config_path)
    values = vars(parser.parse_args(argv))
    values.pop("config", None)
    return TrainConfig(**values)


def _normalize_action_type(action_type: str) -> str:
    normalized = action_type.strip().lower()
    if normalized not in {"policy", "torque"}:
        raise ValueError(f"action_type must be 'policy' or 'torque', got {action_type!r}.")
    return normalized


def resolve_hdf5_paths(cfg: TrainConfig) -> list[str]:
    """Locate the dataset from ``dataset_file``, ``dataset_id``, or ``dataset_dir``.

    ``dataset_file`` is checked first on purpose. Checkpoints written before
    ``dataset_id`` existed carry only a ``dataset_file``, and reloading one
    merges it over current defaults - which do set ``dataset_id``. Letting
    ``dataset_id`` win would silently evaluate such a checkpoint against
    whatever dataset the default config happens to name today.

    ``dataset_id`` names a directory under the data lake's ``raw/`` and resolves
    through that dataset's manifest. All three may use ``${PICR_DATA_ROOT}``.
    """
    if cfg.dataset_file:
        path = os.path.abspath(paths.expand(cfg.dataset_file))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return [path]
    if cfg.dataset_id:
        resolved = paths.dataset_file(cfg.dataset_id)
        if not resolved.is_file():
            raise FileNotFoundError(f"Dataset {cfg.dataset_id!r} resolved to a missing file: {resolved}")
        return [str(resolved)]
    dataset_dir = os.path.abspath(paths.expand(cfg.dataset_dir))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    hdf5_paths = object_mlp_dataset.discover_hdf5_files(dataset_dir)
    if not hdf5_paths:
        raise FileNotFoundError(f"No *.hdf5 files found in {dataset_dir}")
    return hdf5_paths


def _subsample_refs(refs: list[Any], fraction: float, seed: int, kind: str) -> list[Any]:
    if fraction >= 1.0:
        return refs
    if fraction <= 0.0:
        raise ValueError(f"--{kind}_subsample_fraction must be > 0; got {fraction}")
    n = max(1, int(round(len(refs) * fraction)))
    if n >= len(refs):
        return refs
    return random.Random(seed).sample(refs, n)


def make_dataloaders(cfg: TrainConfig):
    hdf5_paths = resolve_hdf5_paths(cfg)
    episode_refs = object_mlp_dataset.load_all_episode_refs(hdf5_paths)
    train_refs = object_mlp_dataset.resolve_split_refs(episode_refs, cfg, "train")
    val_refs = object_mlp_dataset.resolve_split_refs(episode_refs, cfg, "val")
    n_train_before, n_val_before = len(train_refs), len(val_refs)
    if cfg.external_split_json:
        print(
            f"[external split] {cfg.external_split_json}: train {n_train_before} episodes, "
            f"val('{cfg.external_split_val_key}') {n_val_before} episodes "
            "-- train_split/seed do not affect the partition"
        )
    if cfg.train_mass_min > 0.0 or cfg.train_mass_max < float("inf"):
        # Adaptiveness study: train (and select) only on a mass band so the rest of the
        # range stays genuinely unseen. Applied to val too, else model selection leaks.
        train_refs = filter_refs_by_mass(train_refs, cfg.train_mass_min, cfg.train_mass_max)
        val_refs = filter_refs_by_mass(val_refs, cfg.train_mass_min, cfg.train_mass_max)
        print(
            f"[mass filter] keeping [{cfg.train_mass_min}, {cfg.train_mass_max}) kg: "
            f"train {n_train_before} -> {len(train_refs)}, val {n_val_before} -> {len(val_refs)}"
        )
        if not train_refs or not val_refs:
            raise RuntimeError("Mass filter left an empty train or validation set.")
    train_refs = _subsample_refs(train_refs, cfg.train_subsample_fraction, cfg.seed + 1, "train")
    val_refs = _subsample_refs(val_refs, cfg.val_subsample_fraction, cfg.seed + 2, "val")

    layout = object_mlp_dataset.RobotObjectWMStateLayout(
        robot_dof=cfg.robot_dof,
        action_dim=cfg.action_dim,
        torque_dim=cfg.torque_dim,
    )
    dataset_kwargs = {
        "history_len": cfg.history_len,
        "rollout_horizon": cfg.rollout_horizon,
        "dt": cfg.dt,
        "torque_key": cfg.torque_key,
        "filter_pre_contact": cfg.filter_pre_contact,
        "object_displacement_threshold": cfg.object_displacement_threshold,
        "object_velocity_threshold": cfg.object_velocity_threshold,
        "contact_consecutive_steps": cfg.contact_consecutive_steps,
        "contact_settle_steps": cfg.contact_settle_steps,
        "subtract_env_origin": cfg.subtract_env_origin,
        "phase_filter": cfg.phase_filter,
        "filter_effort_saturated": cfg.filter_effort_saturated,
        "filter_velocity_saturated": cfg.filter_velocity_saturated,
        "fd_clock": cfg.fd_clock,
        "context_label_key": cfg.context_label_key,
        "mcg_target_file": paths.expand(cfg.mcg_target_file) if cfg.mcg_target_file else None,
        "recompute_object_ang_vel": cfg.recompute_object_ang_vel,
        "align_object_clock": cfg.align_object_clock,
        "window_stride": cfg.window_stride,
        "history_torque_lag": cfg.history_torque_lag,
        "layout": layout,
    }
    train_dataset = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=train_refs,
        **dataset_kwargs,
    )
    val_dataset = object_mlp_dataset.RobotObjectWMRolloutDataset(
        episode_refs=val_refs,
        **dataset_kwargs,
    )
    meta = {
        "hdf5_paths": hdf5_paths,
        "n_episodes_total": len(episode_refs),
        "n_train_episodes_before_subsample": n_train_before,
        "n_val_episodes_before_subsample": n_val_before,
        "n_train_episodes": len(train_refs),
        "n_val_episodes": len(val_refs),
        "train_subsample_fraction": cfg.train_subsample_fraction,
        "val_subsample_fraction": cfg.val_subsample_fraction,
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
        "torque_key": cfg.torque_key,
        "action_type": cfg.action_type,
        "train_episodes_with_contact": train_dataset.episodes_with_contact,
        "val_episodes_with_contact": val_dataset.episodes_with_contact,
        "train_skipped_short_episodes": train_dataset.skipped_short_episodes,
        "val_skipped_short_episodes": val_dataset.skipped_short_episodes,
        "train_skipped_missing_torque_episodes": train_dataset.skipped_missing_torque_episodes,
        "val_skipped_missing_torque_episodes": val_dataset.skipped_missing_torque_episodes,
        "train_skipped_missing_object_dynamics_episodes": train_dataset.skipped_missing_object_dynamics_episodes,
        "val_skipped_missing_object_dynamics_episodes": val_dataset.skipped_missing_object_dynamics_episodes,
        "train_skipped_contact_filtered_windows": train_dataset.skipped_contact_filtered_windows,
        "val_skipped_contact_filtered_windows": val_dataset.skipped_contact_filtered_windows,
        "phase_filter": cfg.phase_filter,
        "train_skipped_phase_filtered_windows": train_dataset.skipped_phase_filtered_windows,
        "val_skipped_phase_filtered_windows": val_dataset.skipped_phase_filtered_windows,
        "filter_effort_saturated": cfg.filter_effort_saturated,
        "filter_velocity_saturated": cfg.filter_velocity_saturated,
        "fd_clock": cfg.fd_clock,
        "train_skipped_saturated_windows": train_dataset.skipped_saturated_windows,
        "val_skipped_saturated_windows": val_dataset.skipped_saturated_windows,
    }
    if cfg.lambda_context_infonce > 0.0:
        # Label-free InfoNCE needs two windows of the same episode inside a batch; a
        # shuffled loader supplies them essentially never. Batch count is kept at the
        # shuffled loader's value so an epoch still means the same amount of data. The
        # val sampler is deterministic, else model selection compares different draws.
        def _view_loader(dataset, seed: int, deterministic: bool, shrink_to_fit: bool = False):
            batch_size = cfg.batch_size
            if shrink_to_fit:
                # A small val split (e.g. 32 episodes) cannot fill batch_size / 2 episode pairs;
                # shrink the val batch instead of failing. No effect when the split is large enough.
                n_eps = sum(len(idx) >= 2 for idx in dataset.episode_window_indices)
                batch_size = max(4, min(batch_size, 2 * n_eps))
            sampler = EpisodeTwoViewBatchSampler(
                episode_window_indices=dataset.episode_window_indices,
                window_start=dataset.window_start,
                batch_size=batch_size,
                num_batches=max(1, len(dataset) // batch_size),
                min_view_gap=cfg.infonce_min_view_gap,
                seed=seed,
                deterministic=deterministic,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        train_loader = _view_loader(train_dataset, cfg.seed + 3, deterministic=False)
        val_loader = _view_loader(val_dataset, cfg.seed + 4, deterministic=True, shrink_to_fit=True)
        meta["infonce_two_view_batches"] = True
        meta["infonce_min_view_gap"] = cfg.infonce_min_view_gap
        meta["n_train_episodes_with_2plus_windows"] = len(train_loader.batch_sampler.episodes)
        meta["n_val_episodes_with_2plus_windows"] = len(val_loader.batch_sampler.episodes)
        print(
            f"[infonce] two-view batches: {cfg.batch_size // 2} episodes x 2 views, "
            f"min gap {cfg.infonce_min_view_gap}, "
            f"{len(train_loader.batch_sampler.episodes)} train / "
            f"{len(val_loader.batch_sampler.episodes)} val episodes usable"
        )
        return train_loader, val_loader, layout, meta

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, layout, meta


def _max_batches(loader: DataLoader, fraction: float) -> int | None:
    if fraction >= 1.0:
        return None
    if fraction <= 0.0:
        raise ValueError("--train_batch_fraction must be > 0")
    return max(1, int(round(len(loader) * fraction)))


def build_model(
    cfg: TrainConfig,
    *,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    if cfg.model_type == "arm_payload":
        arm_payload_cfg = ArmPayloadWMDynamicsConfig(
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
            delan_eps_diag=cfg.delan_eps_diag,
            delan_parameterize_inverse=cfg.delan_parameterize_inverse,
            qdd_residual=cfg.qdd_residual,
            qdd_residual_hidden_dim=cfg.qdd_residual_hidden_dim,
            qdd_residual_depth=cfg.qdd_residual_depth,
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
        model = build_arm_payload_wm_dynamics(arm_payload_cfg)
        if cfg.arm_prior:
            if not cfg.arm_prior_ckpt:
                raise ValueError(
                    "arm_prior=True needs arm_prior_ckpt: the split only helps if the arm branch "
                    "already carries the logged M, C, G. Fit one with "
                    "python3 -m cdelan.analysis.fit_arm_prior."
                )
            state = torch.load(paths.expand(cfg.arm_prior_ckpt), map_location="cpu",
                               weights_only=False)
            model.core.arm.load_state_dict(state["arm_core"])
            if cfg.arm_prior_freeze:
                model.core.freeze_arm()
            f = state["fit"]
            print(f"[arm prior] {cfg.arm_prior_ckpt}  fit rel_rmse H/c/g = "
                  f"{f['H']:.3f} / {f['c']:.3f} / {f['g']:.3f}   frozen={cfg.arm_prior_freeze}")
        return model
    raise ValueError(
        f"Unknown model_type: {cfg.model_type!r} (this release builds arm_payload only)."
    )
def _forward_model(model, batch, device: torch.device, *, return_context: bool = False):
    history_states = batch["history_states"].to(device)
    future_torques = batch["future_torques"].to(device)
    object_context = batch["object_context"].to(device)
    history_torques = batch["history_torques"].to(device) if getattr(model, "use_context_encoder", False) else None
    return model(
        history_states,
        future_torques,
        object_context,
        history_torques=history_torques,
        return_context=return_context,
    )


@torch.no_grad()
def _integrated_velocity_target(model, history_states, future_states, cfg: "TrainConfig"):
    """future_states with q and the object pose replaced by what the TRUE velocities integrate to.

    q~_h = q_0 + sum of the integrator's own position update fed the recorded velocities, so the
    position channels carry accumulated velocity error and nothing else; the object channels are
    ``assemble_state(q~, dq_true, grasp)`` with the grasp frozen at the anchor, as in the rollout.
    """
    r = cfg.robot_dof
    last = history_states[:, -1]
    dq = future_states[..., r : 2 * r]
    dq_prev = torch.cat([last[:, None, r : 2 * r], dq[:, :-1]], dim=1)
    if cfg.ode_solver in ("trapezoid", "trapezoid_pc", "trapezoid_mid"):
        inc = 0.5 * (dq_prev + dq) * cfg.dt
    elif cfg.ode_solver == "euler":
        inc = dq * cfg.dt  # semi-implicit: q moves with the END velocity
    else:
        raise ValueError(f"rollout_target=integrated_velocity is not defined for ode_solver={cfg.ode_solver!r}")
    q_tilde = last[:, None, :r] + inc.cumsum(dim=1)
    b, h = dq.shape[:2]
    d_off, th_off = model.infer_grasp(last)
    grasp = (d_off.repeat_interleave(h, dim=0), th_off.repeat_interleave(h, dim=0))
    flat = model.assemble_state(q_tilde.reshape(b * h, r), dq.reshape(b * h, r), grasp)
    return flat.reshape(b, h, -1)


def _inverse_dynamics_terms(model, batch, device: torch.device, cfg: "TrainConfig", z):
    """Teacher-forced (H, c, g, qdd, tau) over the rollout window's h frames.

    Frame i of the window is the state the model is asked to step FROM, so its ground
    truth is ``history_states[-1]`` for i = 0 and ``future_states[i-1]`` after that --
    the same indices ``future_torques`` and ``future_qdd`` are cut at. Evaluating at the
    logged (q, dq) rather than the rolled-out ones is deliberate: a drifted state would
    confound "is H right" with "did the trajectory diverge", which is the same reason
    ``analysis/mcg_vs_gt`` teacher-forces.
    """
    r = cfg.robot_dof
    history_states = batch["history_states"].to(device)
    future_states = batch["future_states"].to(device)
    tau = batch["future_torques"].to(device)
    horizon = tau.shape[1]

    frames = torch.cat([history_states[:, -1:, : 2 * r], future_states[:, :-1, : 2 * r]], dim=1)
    q_rel, dq = frames[..., :r], frames[..., r : 2 * r]

    if cfg.id_qdd_source == "fd":
        # What the integrator realises over the interval, not what Isaac saw at the step.
        dq_next = future_states[..., r : 2 * r]
        qdd = (dq_next - dq) / cfg.dt
    else:
        qdd = batch["future_qdd"].to(device)

    flat = lambda x: x.reshape(-1, r)
    z_flat = None if z is None else z.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, z.shape[-1])
    if cfg.ode_solver == "trapezoid_mid":
        # the half-step state the integrator evaluates at (velocity: the interval's true midpoint)
        q_rel = q_rel + 0.5 * cfg.dt * dq
        dq = 0.5 * (dq + future_states[..., r : 2 * r])
    terms = model.inverse_dynamics_tau(flat(q_rel), flat(dq), z=z_flat)
    return terms, flat(qdd), flat(tau)


def _qdd_label_loss(model, batch, device: torch.device, cfg: "TrainConfig", z, qdd_scale):
    """Teacher-forced acceleration-space label loss over the rollout window's h frames.

    Same frames and same z handling as :func:`_inverse_dynamics_terms`; the target is the
    finite-difference qdd_bar = (dq_{j+1} - dq_j) / dt, never the logged instantaneous qdd.
    """
    r, n = cfg.robot_dof, cfg.loss_arm_dof
    history_states = batch["history_states"].to(device)
    future_states = batch["future_states"].to(device)
    tau = batch["future_torques"].to(device)
    horizon = tau.shape[1]
    frames = torch.cat([history_states[:, -1:, : 2 * r], future_states[:, :-1, : 2 * r]], dim=1)
    q_rel, dq = frames[..., :r], frames[..., r : 2 * r]
    target = (future_states[..., r : 2 * r] - dq) / cfg.dt
    flat = lambda x: x.reshape(-1, r)
    z_flat = None if z is None else z.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, z.shape[-1])
    ddq, _ = model.acceleration(flat(q_rel), flat(dq), flat(tau), z=z_flat)
    residual = (ddq - flat(target))[:, :n]
    with torch.no_grad():
        metrics = {"qdd_label_rmse_rad_s2": residual.pow(2).mean().sqrt()}
    if qdd_scale is not None:
        residual = residual / qdd_scale[:n]
    loss = (residual**2).mean()
    metrics["qdd_label_loss"] = loss.detach()
    return loss, metrics


def _rollout_qdd_loss(pred_future, future_states, history_states, cfg: "TrainConfig", qdd_scale):
    """Per-step velocity-difference acceleration of the predicted rollout vs the logged one."""
    r, n = cfg.robot_dof, cfg.loss_arm_dof
    dq0 = history_states[:, -1:, r : 2 * r]
    dq_hat = torch.cat([dq0, pred_future[..., r : 2 * r]], dim=1)
    dq_log = torch.cat([dq0, future_states[..., r : 2 * r]], dim=1)
    a_hat = (dq_hat[:, 1:] - dq_hat[:, :-1]) / cfg.dt
    a_log = (dq_log[:, 1:] - dq_log[:, :-1]) / cfg.dt
    residual = (a_hat - a_log)[..., :n]
    with torch.no_grad():
        metrics = {"rollout_qdd_rmse_rad_s2": residual.pow(2).mean().sqrt()}
    if qdd_scale is not None:
        residual = residual / qdd_scale[:n]
    loss = (residual**2).mean()
    metrics["rollout_qdd_loss"] = loss.detach()
    return loss, metrics


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().detach().cpu().item())


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach()
    finite = torch.isfinite(value)
    nan_count = int(torch.isnan(value).sum().detach().cpu().item())
    inf_count = int(torch.isinf(value).sum().detach().cpu().item())
    finite_count = int(finite.sum().detach().cpu().item())
    total = value.numel()
    if finite_count > 0:
        finite_values = value[finite]
        min_value = float(finite_values.min().detach().cpu().item())
        max_value = float(finite_values.max().detach().cpu().item())
        abs_max = float(finite_values.abs().max().detach().cpu().item())
    else:
        min_value = max_value = abs_max = float("nan")
    return (
        f"{name}: shape={tuple(value.shape)} finite={finite_count}/{total} "
        f"nan={nan_count} inf={inf_count} min={min_value:.6g} max={max_value:.6g} abs_max={abs_max:.6g}"
    )


def _raise_nonfinite_step(phase: str, batch_idx: int, tensors: dict[str, torch.Tensor]) -> None:
    summaries = [_tensor_summary(name, tensor) for name, tensor in tensors.items()]
    joined = "\n  ".join(summaries)
    raise FloatingPointError(f"Non-finite value during {phase} batch {batch_idx}:\n  {joined}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
    max_batches: int | None,
    channel_std: torch.Tensor | None = None,
    tau_scale: torch.Tensor | None = None,
    qdd_scale: torch.Tensor | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    averager = MetricAverager()
    num_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    phase = "train" if is_train else "val"

    # Fixed bounds, so the same theta pair has the same d_Theta in every batch. Built once
    # per epoch rather than per batch, though the tensor is (D, 2).
    contrastive_label_range = (
        torch.as_tensor(cfg.contrastive_label_range, dtype=torch.float32, device=device)
        if cfg.contrastive_label_range is not None
        else None
    )

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        future_torques = batch["future_torques"].to(device)
        future_states = batch["future_states"].to(device)

        with torch.set_grad_enabled(is_train):
            # With lambda_rollout = 0 the rollout is a metric, not an objective. Building
            # its graph would cost h DeLaN evaluations per sample for a term that is
            # multiplied out, so drop the graph but keep the number: val_rollout stays
            # comparable against every other run in the lab.
            rollout_grad = is_train and (cfg.lambda_rollout > 0.0 or cfg.lambda_rollout_qdd > 0.0)
            with torch.set_grad_enabled(rollout_grad):
                if cfg.use_context_encoder:
                    pred_future, context = _forward_model(model, batch, device, return_context=True)
                else:
                    pred_future = _forward_model(model, batch, device)
                    context = None
            if context is not None and not rollout_grad and is_train:
                # The encoder still needs gradients from the KL / contrastive / ID terms.
                context = model.encode_context(
                    batch["history_states"].to(device), batch["history_torques"].to(device)
                )
            rollout_target = future_states
            if cfg.rollout_target == "integrated_velocity":
                rollout_target = _integrated_velocity_target(
                    model, batch["history_states"].to(device), future_states, cfg
                )
            rollout_loss, rollout_metrics = weighted_rollout_mse(
                pred=pred_future,
                target=rollout_target,
                robot_dof=cfg.robot_dof,
                q_weight=cfg.q_weight,
                dq_weight=cfg.dq_weight,
                object_pos_weight=cfg.object_pos_weight,
                object_quat_weight=cfg.object_quat_weight,
                object_lin_vel_weight=cfg.object_lin_vel_weight,
                object_ang_vel_weight=cfg.object_ang_vel_weight,
                channel_std=channel_std,
                arm_dof=cfg.loss_arm_dof,
            )
            if cfg.rollout_target != "logged":
                # the same number every other run reports as val_rollout, so curves stay comparable
                with torch.no_grad():
                    logged_loss, _ = weighted_rollout_mse(
                        pred=pred_future.detach(),
                        target=future_states,
                        robot_dof=cfg.robot_dof,
                        q_weight=cfg.q_weight,
                        dq_weight=cfg.dq_weight,
                        object_pos_weight=cfg.object_pos_weight,
                        object_quat_weight=cfg.object_quat_weight,
                        object_lin_vel_weight=cfg.object_lin_vel_weight,
                        object_ang_vel_weight=cfg.object_ang_vel_weight,
                        channel_std=channel_std,
                        arm_dof=cfg.loss_arm_dof,
                    )
                rollout_metrics = dict(rollout_metrics)
                rollout_metrics["rollout_loss_logged"] = logged_loss
            if context is not None:
                context_loss, context_metrics = context_kl_loss(context, cfg.lambda_context_kl)
            else:
                context_loss = rollout_loss.new_zeros(())
                context_metrics = {}
            mass_loss = rollout_loss.new_zeros(())
            if (
                context is not None
                and getattr(context, "mass_pred", None) is not None
                and cfg.lambda_mass_aux > 0.0
            ):
                true_mass = batch["object_context"][:, 0:1].to(device)
                mass_loss = ((context.mass_pred - true_mass) ** 2).mean()
                context_metrics = dict(context_metrics)
                context_metrics["mass_mse"] = mass_loss.detach()
                context_metrics["mass_mae"] = (context.mass_pred - true_mass).abs().mean().detach()
            contrastive_loss = rollout_loss.new_zeros(())
            if context is not None and cfg.lambda_context_contrastive > 0.0:
                # The axis the latent is asked to be proportional to. "mass" reproduces
                # the original behaviour; the damping datasets structure z by the arm's
                # PD damping gain instead, which is what varies per episode there.
                mass_target = batch["context_label"].reshape(batch["context_label"].shape[0], -1).to(device)
                contrastive_loss, contrastive_metrics = context_contrastive_loss(
                    context,
                    mass_target,
                    weight=cfg.lambda_context_contrastive,
                    temperature=cfg.contrastive_temperature,
                    mass_delta=cfg.contrastive_mass_delta,
                    mode=cfg.contrastive_mode,
                    per_factor=cfg.contrastive_per_factor,
                    label_norm=cfg.contrastive_label_norm,
                    label_range=contrastive_label_range,
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(contrastive_metrics)
            infonce_loss = rollout_loss.new_zeros(())
            if context is not None and cfg.lambda_context_infonce > 0.0:
                nce_context = context
                if cfg.infonce_latent_dims > 0:
                    # contrast only the leading block of z; the rest is shaped by the rollout alone
                    from dataclasses import replace as _replace
                    nce_context = _replace(context, mu=context.mu[:, : cfg.infonce_latent_dims])
                infonce_loss, infonce_metrics = context_infonce_loss(
                    nce_context,
                    batch["episode_id"],
                    weight=cfg.lambda_context_infonce,
                    temperature=cfg.infonce_temperature,
                    positives=cfg.infonce_positives,
                    label=None if cfg.infonce_positives == "episode" else batch["context_label"],
                    mass_delta=cfg.infonce_mass_delta,
                    com_sigma=cfg.infonce_com_sigma,
                    mass_sigma=cfg.infonce_mass_sigma,
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(infonce_metrics)
            id_loss = rollout_loss.new_zeros(())
            if cfg.lambda_inverse_dynamics > 0.0:
                terms, qdd_flat, tau_flat = _inverse_dynamics_terms(
                    model, batch, device, cfg, None if context is None else context.z
                )
                id_loss, id_metrics = inverse_dynamics_loss(
                    terms.H,
                    terms.coriolis,
                    terms.g,
                    qdd_flat,
                    tau_flat,
                    arm_dof=cfg.id_loss_arm_dof,
                    tau_scale=tau_scale,
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(id_metrics)
            hist_id_loss = rollout_loss.new_zeros(())
            if cfg.lambda_history_id > 0.0 and context is not None:
                n = cfg.id_loss_arm_dof
                hs_ = batch["history_states"].to(device)
                # a few random intervals per step: all K-1 would cost more than the rollout itself
                pick = torch.randperm(hs_.shape[1] - 1, device=device)[: max(1, cfg.history_id_intervals)]
                hres = model.history_residual(
                    hs_, batch["history_torques"].to(device), z=context.z, intervals=pick
                )[..., :n]
                if tau_scale is not None:
                    hres = hres / tau_scale[:n]
                hist_id_loss = (hres ** 2).mean()
                context_metrics = dict(context_metrics)
                context_metrics["history_id_loss"] = hist_id_loss.detach()
            mcg_loss = rollout_loss.new_zeros(())
            if cfg.lambda_mcg > 0.0:
                # Same teacher-forced frames the ID term uses, so the two arms of the ablation
                # are scored on identical states.
                mterms, _, _ = _inverse_dynamics_terms(
                    model, batch, device, cfg, None if context is None else context.z
                )
                flat7 = lambda x: x.reshape(-1, *x.shape[2:])
                mcg_loss, mcg_metrics = mcg_supervision_loss(
                    mterms.H,
                    mterms.coriolis,
                    mterms.g,
                    flat7(batch["future_mcg_H"].to(device)),
                    flat7(batch["future_mcg_c"].to(device)),
                    flat7(batch["future_mcg_g"].to(device)),
                    arm_dof=cfg.mcg_loss_arm_dof,
                    normalize=cfg.mcg_loss_normalize,
                    weights=(cfg.mcg_weight_H, cfg.mcg_weight_c, cfg.mcg_weight_g),
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(mcg_metrics)
            qdd_label_loss = rollout_loss.new_zeros(())
            if cfg.lambda_qdd_label > 0.0:
                qdd_label_loss, qdd_label_metrics = _qdd_label_loss(
                    model, batch, device, cfg, None if context is None else context.z, qdd_scale
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(qdd_label_metrics)
            rollout_qdd_loss = rollout_loss.new_zeros(())
            if cfg.lambda_rollout_qdd > 0.0:
                rollout_qdd_loss, rollout_qdd_metrics = _rollout_qdd_loss(
                    pred_future, future_states, batch["history_states"].to(device), cfg, qdd_scale
                )
                context_metrics = dict(context_metrics)
                context_metrics.update(rollout_qdd_metrics)
            loss = (
                cfg.lambda_rollout * rollout_loss
                + context_loss + cfg.lambda_mass_aux * mass_loss
                + contrastive_loss + infonce_loss
                + cfg.lambda_inverse_dynamics * id_loss
                + cfg.lambda_history_id * hist_id_loss
                + cfg.lambda_mcg * mcg_loss
                + cfg.lambda_qdd_label * qdd_label_loss
                + cfg.lambda_rollout_qdd * rollout_qdd_loss
            )
            pop_aux = getattr(model, "pop_aux_loss", None)
            aux_loss = pop_aux() if pop_aux is not None else None
            if aux_loss is not None and is_train:
                # model-specific auxiliary training term; train only
                loss = loss + aux_loss
            if not _is_finite(loss):
                debug_tensors = {
                    "history_states": batch["history_states"].to(device),
                    "history_torques": batch["history_torques"].to(device),
                    "future_torques": future_torques,
                    "future_states": future_states,
                    "pred_future": pred_future,
                    "rollout_loss": rollout_loss,
                    "context_loss": context_loss,
                    "loss": loss,
                }
                if context is not None:
                    debug_tensors.update(context.as_aux())
                debug_tensors.update(rollout_metrics)
                debug_tensors.update(context_metrics)
                _raise_nonfinite_step(phase, batch_idx, debug_tensors)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=getattr(model, "grad_clip_norm", 10.0)
            )
            if not _is_finite(grad_norm):
                _raise_nonfinite_step(phase, batch_idx, {"loss": loss, "grad_norm": grad_norm})
            optimizer.step()

        metrics = _step_metrics(
            loss=loss,
            rollout_loss=rollout_loss,
            pred_future=pred_future,
            future_states=future_states,
            future_torques=future_torques,
            robot_dof=cfg.robot_dof,
        )
        metrics.update(rollout_metrics)
        if context_metrics:
            metrics.update(context_metrics)
            metrics["context_kl_loss"] = context_loss.detach()
        if cfg.lambda_inverse_dynamics > 0.0:
            metrics["id_loss"] = id_loss.detach()
        if cfg.lambda_qdd_label > 0.0:
            metrics["qdd_label_loss"] = qdd_label_loss.detach()
        if cfg.lambda_rollout_qdd > 0.0:
            metrics["rollout_qdd_loss"] = rollout_qdd_loss.detach()
        if cfg.lambda_mcg > 0.0:
            metrics["mcg_loss"] = mcg_loss.detach()
        averager.update(metrics)
        averager.step()

        if cfg.log_every_batches > 0 and batch_idx % cfg.log_every_batches == 0:
            print(f"  [{phase}] batch {batch_idx}/{num_batches} loss={loss.item():.6f}")

    return averager.means()


def _step_metrics(
    *,
    loss,
    rollout_loss,
    pred_future,
    future_states,
    future_torques,
    robot_dof: int,
) -> dict[str, torch.Tensor]:
    robot_state_dim = 2 * robot_dof
    one_step_q = torch.nn.functional.mse_loss(pred_future[:, 0, :robot_dof], future_states[:, 0, :robot_dof])
    one_step_object_pos = torch.nn.functional.mse_loss(
        pred_future[:, 0, robot_state_dim : robot_state_dim + 3],
        future_states[:, 0, robot_state_dim : robot_state_dim + 3],
    )
    final_step_q = torch.nn.functional.mse_loss(pred_future[:, -1, :robot_dof], future_states[:, -1, :robot_dof])
    final_step_object_pos = torch.nn.functional.mse_loss(
        pred_future[:, -1, robot_state_dim : robot_state_dim + 3],
        future_states[:, -1, robot_state_dim : robot_state_dim + 3],
    )
    metrics = {
        "loss": loss.detach(),
        "rollout_loss": rollout_loss.detach(),
        "one_step_q_mse": one_step_q.detach(),
        "one_step_object_pos_mse": one_step_object_pos.detach(),
        "final_step_q_mse": final_step_q.detach(),
        "final_step_object_pos_mse": final_step_object_pos.detach(),
        "torque_abs_mean": future_torques.abs().mean().detach(),
    }
    return metrics


def _print_run_header(
    cfg: TrainConfig,
    device: torch.device,
    output_dir: str,
    meta: dict[str, Any],
    *,
    model_name: str = MODEL_NAME,
) -> None:
    print("Training config:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print("Dataset meta:")
    for key, value in meta.items():
        print(f"  {key}: {value}")


def _optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _post_training_eval_output_dir(cfg: TrainConfig, run_output_dir: str) -> str:
    if os.path.isabs(cfg.eval_output_dir):
        return cfg.eval_output_dir
    return os.path.join(run_output_dir, cfg.eval_output_dir)


def _metric_key(value: str) -> str:
    return value.replace(os.sep, "/").replace(" ", "_")


def _flatten_eval_metrics(prefix: str, value: Any, metrics: dict[str, float]) -> None:
    if isinstance(value, bool):
        metrics[prefix] = float(value)
    elif isinstance(value, (int, float)):
        metrics[prefix] = float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}/{_metric_key(str(key))}" if prefix else _metric_key(str(key))
            _flatten_eval_metrics(next_prefix, item, metrics)
    elif isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        metrics[f"{prefix}/mean"] = float(sum(value) / len(value))
        metrics[f"{prefix}/final"] = float(value[-1])


def _read_eval_metrics(output_dir: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    json_specs = {
        "summary.json": "summary",
        "episode_metrics.json": "episode",
        "prediction_metrics.json": "prediction",
    }
    for filename, prefix in json_specs.items():
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            _flatten_eval_metrics(prefix, payload, metrics)
        except Exception as exc:
            print(f"[WARN] Could not read eval metrics from {path}: {exc}")
    return metrics


def _collect_eval_media(output_dir: str, written: dict[str, str] | None = None) -> dict[str, str]:
    media_exts = {".png", ".jpg", ".jpeg", ".mp4", ".mov", ".m4v", ".gif"}
    media: dict[str, str] = {}
    seen_paths: set[str] = set()
    for key, path in (written or {}).items():
        if os.path.isfile(path) and Path(path).suffix.lower() in media_exts:
            seen_paths.add(os.path.abspath(path))
            media[_metric_key(str(key))] = path
    if not os.path.isdir(output_dir):
        return media
    for root, _, files in os.walk(output_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if Path(path).suffix.lower() not in media_exts:
                continue
            abs_path = os.path.abspath(path)
            if abs_path in seen_paths:
                continue
            rel = os.path.relpath(path, output_dir)
            key = _metric_key(os.path.splitext(rel)[0])
            media.setdefault(key, path)
            seen_paths.add(abs_path)
    return media


def run_post_training_evaluation(cfg: TrainConfig, run) -> None:
    if not cfg.eval_after_train:
        return

    output_dir = _post_training_eval_output_dir(cfg, run.output_dir)
    argv = [
        "--checkpoint",
        run.best_path,
        "--output_dir",
        output_dir,
        "--split",
        cfg.eval_split,
        "--num_workers",
        str(cfg.eval_num_workers),
        "--episode_index",
        str(cfg.eval_episode_index),
        "--start_t",
        str(cfg.eval_start_t),
        "--rollout_steps",
        str(cfg.eval_rollout_steps),
        "--target",
        cfg.eval_target,
        "--pred_horizon",
        str(cfg.eval_pred_horizon),
        "--prediction_max_episodes",
        str(cfg.eval_prediction_max_episodes),
        "--fps",
        str(cfg.eval_fps),
        "--max_frames",
        str(cfg.eval_max_frames),
    ]
    if cfg.dataset_file:
        argv.extend(["--dataset_file", cfg.dataset_file])
    else:
        argv.extend(["--dataset_dir", cfg.dataset_dir])
    _optional_arg(argv, "--batch_size", cfg.eval_batch_size)
    _optional_arg(argv, "--max_batches", cfg.eval_max_batches)
    _optional_arg(argv, "--max_episodes", cfg.eval_max_episodes)
    _optional_arg(argv, "--episode_name", cfg.eval_episode_name)
    if cfg.eval_episode_plot:
        argv.append("--episode_plot")
    if cfg.eval_prediction_metrics:
        argv.append("--prediction_metrics")
    if cfg.eval_video:
        argv.append("--video")

    print("===== Post-training evaluation =====")
    print(f"Best checkpoint: {run.best_path}")
    print(f"Eval output dir: {output_dir}")
    try:
        from cdelan.eval.rollout import main as eval_main

        written = eval_main(argv)
        metrics = _read_eval_metrics(output_dir)
        media = _collect_eval_media(output_dir, written)
        if not media:
            print(f"[WARN] No eval media files found under {output_dir}.")
        run.log_evaluation(output_dir, metrics=metrics, media=media, fps=cfg.eval_fps)
    except Exception as exc:
        if cfg.eval_fail_on_error:
            raise
        print(f"[WARN] Post-training evaluation failed: {exc}")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    cfg.action_type = _normalize_action_type(cfg.action_type)
    model_name = MODEL_NAME
    run_config = asdict(cfg)

    set_seed(cfg.seed)
    run = start_training_run(cfg, model_name, run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, layout, meta = make_dataloaders(cfg)
    model = build_model(cfg, device=device).to(device)
    if hasattr(model, "fit_input_stats"):
        model.fit_input_stats(train_loader.dataset)
    # `weight_decay` was a declared-but-unused field until every run before that
    # trained at 0 regardless of its config. It matters here because the Euler-Lagrange
    # equation is invariant to `L' = aL + b` (DeLaN paper eq. 9), and the paper mitigates
    # that non-uniqueness with exactly this L2 term.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    _print_run_header(cfg, device, run.output_dir, meta, model_name=model_name)
    run.update_config({"architecture": model_name, "data_meta": meta})

    best_val = float("inf")
    max_train_batches = _max_batches(train_loader, cfg.train_batch_fraction)

    # Continue a finished run CDELAN_RESUME_FROM=<ckpt> restores the model AND the
    # optimizer state and starts at the checkpoint's epoch + 1, up to cfg.epochs. best.pt of the new
    # run dir is the best of the continued epochs only. The shuffle order is not the one an
    # uninterrupted run would have drawn.
    ema = None
    if cfg.ema_decay > 0.0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
        # a post-step hook, not a wrapped .step: lr schedulers patch the bound method themselves
        optimizer.register_step_post_hook(lambda _opt, _args, _kwargs: ema.update_parameters(model))
        print(f"[ema] weight EMA on, decay {cfg.ema_decay}; validation and checkpoints use the averaged weights")
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.1 * cfg.lr)
    start_epoch = 1
    resume_from = __import__("os").environ.get("CDELAN_RESUME_FROM")
    if resume_from:
        resume_ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        if scheduler is not None:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"[resume] {resume_from}: epoch {resume_ckpt['epoch']} -> continuing at {start_epoch}..{cfg.epochs}")

    channel_std = None
    if cfg.loss_normalize_channels:
        # Fitted on the train split's target trajectories; the floor guards near-constant
        # channels (a pinned finger) from blowing the scale up.
        flat = train_loader.dataset.future_states.reshape(-1, layout.state_dim)
        std = torch.from_numpy(flat.std(axis=0)).float().clamp_min(1e-3).to(device)
        channel_std = std
        print("[loss] per-channel std normalization on; std range "
              f"[{float(std.min()):.4g}, {float(std.max()):.4g}]")

    tau_scale = None
    if cfg.lambda_mcg > 0.0:
        if cfg.model_type != "arm_payload":
            raise ValueError("lambda_mcg requires model_type='arm_payload'.")
        if not cfg.mcg_target_file:
            raise ValueError(
                "lambda_mcg needs --mcg_target_file: the GT_full sidecar written by "
                "python3 -m cdelan.analysis.build_mcg_targets."
            )
        if not getattr(train_loader.dataset, "has_mcg", False):
            raise RuntimeError(
                "lambda_mcg is set but the MCG sidecar did not cover every training episode; "
                "rebuild it from the same hdf5 this run reads."
            )
    if cfg.history_torque_lag and (cfg.lambda_history_id > 0.0 or cfg.context_residual_features
                                   or cfg.context_ls_refine):
        # these pair history_torques[:, :-1] with the history's state intervals, which a lag misaligns
        raise ValueError("history_torque_lag > 0 is not supported with lambda_history_id / "
                         "context_residual_features / context_ls_refine.")
    if cfg.lambda_history_id > 0.0:
        if cfg.model_type != "arm_payload":
            raise ValueError("lambda_history_id requires model_type='arm_payload'.")
        n = cfg.id_loss_arm_dof
        rms = np.sqrt((train_loader.dataset.future_torques[..., :n].astype(np.float64) ** 2).mean(axis=(0, 1)))
        if cfg.id_loss_normalize:
            tau_scale = torch.from_numpy(rms).float().clamp_min(1e-2).to(device)
        print(f"[loss] history-consistency torque loss on (lambda={cfg.lambda_history_id}, "
              f"joints=0..{n - 1}, normalize={cfg.id_loss_normalize})")
    if cfg.lambda_inverse_dynamics > 0.0:
        if cfg.model_type != "arm_payload":
            raise ValueError("lambda_inverse_dynamics requires model_type='arm_payload'.")
        if cfg.id_qdd_source == "logged" and not getattr(train_loader.dataset, "has_qdd", False):
            raise ValueError(
                "lambda_inverse_dynamics with id_qdd_source='logged' needs robot_dynamics/qdd "
                "in every episode; this dataset is missing it. Use --id_qdd_source fd."
            )
        n = cfg.id_loss_arm_dof
        rms = np.sqrt((train_loader.dataset.future_torques[..., :n].astype(np.float64) ** 2).mean(axis=(0, 1)))
        if cfg.id_loss_normalize:
            tau_scale = torch.from_numpy(rms).float().clamp_min(1e-2).to(device)
        print(
            f"[loss] inverse dynamics on (lambda={cfg.lambda_inverse_dynamics}, "
            f"qdd={cfg.id_qdd_source}, joints=0..{n - 1}, normalize={cfg.id_loss_normalize}); "
            f"train tau rms per joint {np.array2string(rms, precision=2)}"
        )
    qdd_scale = None
    if cfg.lambda_qdd_label > 0.0 or cfg.lambda_rollout_qdd > 0.0:
        r, n = cfg.robot_dof, cfg.loss_arm_dof
        ds_ = train_loader.dataset
        dq_seq = np.concatenate(
            [ds_.history_states[:, -1:, r : 2 * r], ds_.future_states[..., r : 2 * r]], axis=1
        ).astype(np.float64)
        label_std = (np.diff(dq_seq, axis=1) / cfg.dt)[..., :n].reshape(-1, n).std(axis=0)
        if cfg.qdd_label_normalize:
            qdd_scale = torch.from_numpy(label_std).float().clamp_min(1e-2).to(device)
        print(
            f"[loss] qdd label scale on (lambda_qdd_label={cfg.lambda_qdd_label}, lambda_rollout_qdd={cfg.lambda_rollout_qdd}, joints=0..{n - 1}, "
            f"normalize={cfg.qdd_label_normalize}); train label std per joint [rad/s^2] "
            f"{np.array2string(label_std, precision=2)}"
        )
    if cfg.lambda_rollout <= 0.0 and cfg.lambda_rollout_qdd <= 0.0:
        print("[loss] rollout weight is 0 -- the state prediction is a metric only.")
    elif cfg.lambda_rollout <= 0.0:
        print("[loss] rollout state loss off; training on the rolled-out velocity-difference qdd (lambda_rollout_qdd).")

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, optimizer, device, cfg, max_train_batches,
                channel_std=channel_std, tau_scale=tau_scale, qdd_scale=qdd_scale,
            )
            eval_model = ema.module if ema is not None else model
            val_metrics = run_epoch(
                eval_model, val_loader, None, device, cfg, None, channel_std=channel_std,
                tau_scale=tau_scale, qdd_scale=qdd_scale,
            )
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_rollout={val_metrics['rollout_loss']:.6f}"
                + (f" val_rollout_logged={val_metrics['rollout_loss_logged']:.6f}"
                   if "rollout_loss_logged" in val_metrics else "")
            )
            best_val = checkpoint_and_log_epoch(
                run=run,
                epoch=epoch,
                checkpoint={
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "architecture": model_name,
                    "config": asdict(cfg),
                    "layout": layout.to_dict(),
                    "data_meta": meta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss=best_val,
                select_key=(
                    "loss" if cfg.best_metric == "loss"
                    else ("rollout_loss_logged" if "rollout_loss_logged" in val_metrics else "rollout_loss")
                ),
            )
            if scheduler is not None:
                scheduler.step()
        print(f"Best checkpoint: {run.best_path}")
        print(f"Last checkpoint: {run.last_path}")
        print(f"W&B run: {run.wandb_url or cfg.wandb_mode}")
        run_post_training_evaluation(cfg, run)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
