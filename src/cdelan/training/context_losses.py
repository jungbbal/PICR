"""Losses that shape the context latent, and the weighted rollout objective.

Lifted verbatim out of the old ``models/world_model.py``, which mixed these with a
15-DOF model this package does not carry. Nothing here depends on a particular
dynamics structure -- they take a :class:`ContextOutput` and a rollout tensor -- so
they belong next to the trainer rather than inside one model file.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.context import ContextOutput, kl_divergence_standard_normal


def context_kl_loss(context: ContextOutput, weight: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if weight <= 0.0 or context.mu is None or context.logvar is None:
        if context.mu is not None:
            zero = context.mu.new_zeros(())
        else:
            zero = torch.zeros(())
        return zero, {}
    kl = kl_divergence_standard_normal(context.mu, context.logvar)
    return weight * kl, {"context_kl": kl.detach()}


def context_infonce_loss(
    context: ContextOutput,
    episode_id: torch.Tensor,
    weight: float = 0.0,
    temperature: float = 0.1,
    positives: str = "episode",
    label: torch.Tensor | None = None,
    mass_delta: float = 0.15,
    com_sigma: float = 0.01,
    mass_sigma: float = 0.15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """InfoNCE (NT-Xent) on ``mu`` over the two-view batches.

    ``positives="mass_com_gauss"`` user): symmetric Gaussian kernel on BOTH factors,

        w_ij = exp(-(m_i - m_j)^2 / (2 mass_sigma^2) - ||r_i - r_j||^2 / (2 com_sigma^2)),  j != i,

    no hard mass mask and no same-episode override (identical labels already give w = 1).
    Loss = -1/M sum_i sum_{j!=i} wbar_ij log softmax_{k!=i}(s_ik), wbar_ij = w_ij / sum_{k!=i} w_ik.

    ``positives="episode"`` (the original, label-free form): positives are the other window(s) of
    the same episode in the batch; every other window is a negative. The payload mass never enters
    the loss -- the only thing asserted is "these two 12-step windows came from the same rollout".

    ``positives="mass_com_soft"`` user: "같은 에피소드끼리 뭉치는걸 넘어서 같은 질량, CoM 상황끼리
    뭉치도록"): supervised, soft-weighted positives from the physical label ``label = [mass, com_x, com_y,
    com_z]`` (kg, m). Window j is a positive of anchor i with weight

        w_ij = 1[|m_i - m_j| <= mass_delta] * exp(-||c_i - c_j||^2 / (2 com_sigma^2)),

    so the same episode (identical label) has weight 1, another episode of the same mass level pulls
    with a weight that decays with its CoM distance (one lattice step, 0.98 cm, -> 0.62 at the default
    com_sigma 1 cm), and a different mass level is a plain negative. The loss is SupCon with these
    weights: -sum_j w_ij log p_ij / sum_j w_ij, with p the softmax over all other windows. The hard
    same-level variant is ``positives="mass"`` (com_sigma ignored, w = 1[same level]).
    """
    if weight <= 0.0 or context.mu is None:
        zero = context.mu.new_zeros(()) if context.mu is not None else torch.zeros(())
        return zero, {}
    mu = context.mu
    b = mu.shape[0]
    if b < 2:
        return mu.new_zeros(()), {}

    z = F.normalize(mu, dim=-1)
    sim = (z @ z.t()) / temperature
    eye = torch.eye(b, device=mu.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)  # drop self from the partition function
    episode_id = episode_id.reshape(-1).to(mu.device)
    same_episode = (episode_id.unsqueeze(1) == episode_id.unsqueeze(0)) & (~eye)
    if positives == "episode":
        w = same_episode.float()
    elif positives in ("mass", "mass_com_soft"):
        if label is None:
            raise ValueError(f"infonce positives={positives!r} needs the context label (mass [, com])")
        lab = label.reshape(b, -1).to(mu.device).float()
        m = lab[:, 0]
        same_mass = ((m.unsqueeze(1) - m.unsqueeze(0)).abs() <= mass_delta) & (~eye)
        w = same_mass.float()
        if positives == "mass_com_soft" and lab.shape[1] > 1:
            c = lab[:, 1:]
            d2 = torch.cdist(c, c).pow(2)
            w = w * torch.exp(-d2 / (2.0 * float(com_sigma) ** 2))
        w = torch.maximum(w, same_episode.float())  # the same episode is always a full positive
    elif positives == "mass_com_gauss":
        if label is None:
            raise ValueError("infonce positives='mass_com_gauss' needs the context label (mass, com)")
        lab = label.reshape(b, -1).to(mu.device).float()
        m = lab[:, 0]
        e = (m.unsqueeze(1) - m.unsqueeze(0)).pow(2) / (2.0 * float(mass_sigma) ** 2)
        if lab.shape[1] > 1:
            c = lab[:, 1:]
            e = e + torch.cdist(c, c).pow(2) / (2.0 * float(com_sigma) ** 2)
        w = torch.exp(-e).masked_fill(eye, 0.0)
    else:
        raise ValueError(f"unknown infonce positives mode {positives!r}")
    wsum = w.sum(1)
    valid = wsum > 0
    if not bool(valid.any()):
        return mu.new_zeros(()), {}
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_anchor = -(logp * w).sum(1) / wsum.clamp_min(1e-12)
    loss = per_anchor[valid].mean()
    with torch.no_grad():
        # top-1 retrieval: does the anchor's nearest neighbour come from its own episode?
        top1 = sim.argmax(dim=1)
        acc = same_episode[torch.arange(b, device=mu.device), top1][valid].float().mean()
    metrics = {
        "infonce": loss.detach(),
        "infonce_top1_acc": acc.detach(),
        "infonce_anchors_with_pos": valid.float().mean().detach(),
    }
    if positives != "episode":
        with torch.no_grad():
            metrics["infonce_pos_weight_per_anchor"] = wsum[valid].mean()
            metrics["infonce_pos_count_per_anchor"] = (w > 0.05).float().sum(1)[valid].mean()
            # does the anchor's nearest neighbour share its mass level?
            if positives == "mass_com_gauss":
                same_lvl = ((m.unsqueeze(1) - m.unsqueeze(0)).abs() <= 0.05)
                metrics["infonce_top1_same_mass"] = same_lvl[torch.arange(b, device=mu.device), top1][valid].float().mean()
            else:
                metrics["infonce_top1_same_mass"] = (w[torch.arange(b, device=mu.device), top1] > 0)[valid].float().mean()
    return weight * loss, metrics


def _standardized_label(label: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(B,) or (B, D) -> (B, D) with each dimension divided by its in-batch std.

    Equalizes axes recorded in different units before a Euclidean gap is taken. A
    near-constant dimension (std ~ 0) is left alone rather than blown up.
    """
    label = label.to(device)
    label = label.reshape(label.shape[0], -1).float()
    if label.shape[1] == 1:
        return label
    std = label.std(dim=0, keepdim=True)
    return label / std.clamp_min(1e-6)


def _range_normalized_label(
    label: torch.Tensor, device: torch.device, bounds: torch.Tensor
) -> torch.Tensor:
    """(B,) or (B, D) -> (B, D) mapped onto [0, 1] by each coordinate's ADMISSIBLE range.

    ``bounds`` is (D, 2) = (theta_min, theta_max), a property of the physical parameter
    space, not of the batch:

        theta~_k = (theta_k - theta_k^min) / (theta_k^max - theta_k^min)

    The alternative, ``_standardized_label``, divides by the in-batch std, which makes the
    metric a property of the sampling distribution: an axis the recording happened to visit
    narrowly is silently blown up until its gaps match mass's. On the 8 x 31 mass-CoM grid
    the two differ by 1.8x -- the CoM axes' std/range is 0.179 against mass's 0.326, because
    21 of the 31 CoM cells hold c_x = 0 exactly -- so std normalization hands each CoM axis
    1.8x the weight the physical range gives it.

    Range normalization is also the only one of the two that is *fixed*: the same theta pair
    has the same distance in every batch, in every run, and on a dataset with a different
    payload distribution. That is what makes d_Theta a metric the encoder can be asked to
    reproduce, rather than a per-batch rescaling.
    """
    label = label.to(device)
    label = label.reshape(label.shape[0], -1).float()
    bounds = bounds.to(device=device, dtype=label.dtype)
    if bounds.shape != (label.shape[1], 2):
        raise ValueError(
            f"contrastive_label_range must be (D, 2) for a D={label.shape[1]} label, "
            f"got {tuple(bounds.shape)}."
        )
    span = (bounds[:, 1] - bounds[:, 0]).clamp_min(1e-12)
    return (label - bounds[:, 0]) / span


def _proportional_rsa_loss(mu: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """1 - corr(pairwise cosine distance of ``mu``, pairwise gap of ``label``).

    Returns ``None`` when the batch carries no spread in the label, so the caller can skip
    that factor for this batch instead of pushing a meaningless gradient.
    """
    b = mu.shape[0]
    eye = torch.eye(b, device=mu.device, dtype=torch.bool)
    z = F.normalize(mu, dim=-1)
    dz = 1.0 - (z @ z.t())
    dm = torch.cdist(label, label)
    off = ~eye
    dz_v, dm_v = dz[off], dm[off]
    if dm_v.std() < 1e-6:
        return None
    dz_c = dz_v - dz_v.mean()
    dm_c = dm_v - dm_v.mean()
    corr = (dz_c * dm_c).sum() / (dz_c.norm() * dm_c.norm() + 1e-8)
    return 1.0 - corr, corr


def context_contrastive_loss(
    context: ContextOutput,
    mass: torch.Tensor,
    weight: float = 0.0,
    temperature: float = 0.1,
    mass_delta: float = 0.15,
    mode: str = "supcon",
    label_norm: str = "std",
    label_range: torch.Tensor | None = None,
    per_factor: str = "none",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Structure the context mean ``mu`` by payload mass. Two modes:

    ``supcon`` (default): supervised-contrastive. Windows whose masses are within
    ``mass_delta`` (kg) are positives (should have high cosine similarity); every
    other in-batch window is a negative. Only "same-ish mass -> same z" is
    supervised, with no mass *value* target. This gives good local clustering but
    does not force the latent distance to grow monotonically with the mass gap, so
    the 1-D mass gradient tends to curl into a horseshoe at the extremes.

    ``proportional``: force the latent cosine distance to be *proportional* to the
    mass gap. We maximize the (differentiable, RSA-style) Pearson correlation
    between the batch's pairwise cosine distances ``D^z_ij = 1 - cos(z_i, z_j)``
    and the pairwise mass distances ``D^m_ij = |m_i - m_j|``. Correlation is
    scale-free (no cosine-scale target to hand-pick) and, being linear, enforces a
    monotone distance->mass relationship — removing the horseshoe curl. The loss
    is ``1 - corr`` so a perfect match gives 0.

    ``mass`` may be (B,) or (B, D). For D > 1 the gap is the Euclidean distance over
    per-dimension standardized coordinates, so an axis recorded in small units (rotor
    armature, ~0.1) is weighted the same as one recorded in large units (joint friction,
    ~1.0) instead of being drowned out. Standardization is a positive rescaling, so for
    D = 1 it leaves the scale-free correlation exactly unchanged.
    """
    if weight <= 0.0 or context.mu is None:
        zero = context.mu.new_zeros(()) if context.mu is not None else torch.zeros(())
        return zero, {}
    mu = context.mu
    b = mu.shape[0]
    if b < 2:
        return mu.new_zeros(()), {}

    if mode == "proportional":
        # How the D physical coordinates are put on a common scale before the Euclidean gap.
        # "std" (the ladder's original behaviour) divides by the in-batch std; "range" maps
        # each onto [0, 1] by its admissible range, which makes d_Theta a fixed metric on the
        # parameter space instead of a per-batch one. See _range_normalized_label.
        if label_norm == "range":
            if label_range is None:
                raise ValueError("label_norm='range' needs contrastive_label_range.")
            mass = _range_normalized_label(mass, mu.device, label_range)
        elif label_norm == "std":
            mass = _standardized_label(mass, mu.device)           # (B, D)
        else:
            raise ValueError(f"label_norm must be 'std' or 'range', got {label_norm!r}.")
        n_factors = mass.shape[1]

        if per_factor in ("blocks", "shared") and n_factors > 1:
            # One contrastive loss per randomized property, instead of one loss against a
            # merged gap. Merging cannot separate the factors: it compares a single scalar
            # cosine distance against a single scalar gap, and for independent factors
            # dm^2 = dm_1^2 + dm_2^2, so one dz correlates only partially with each. That run
            # gave mass rho 0.96 and alpha rho 0.01.
            #
            # "blocks": each factor gets its own disjoint slice of the latent, so each has its
            #   own degrees of freedom and they cannot compete.
            # "shared": every factor's loss is applied to the WHOLE latent. The terms are
            #   summed rather than merged into one gap, so each factor pulls on the shared
            #   geometry with equal weight and neither can be ignored for free -- but they do
            #   share the same degrees of freedom, which at latent_dim 2 means a single angle.
            if per_factor == "blocks":
                if mu.shape[1] < 2 * n_factors:
                    raise ValueError(
                        f"per_factor='blocks' needs latent_dim >= 2 per factor "
                        f"({2 * n_factors} for {n_factors} factors), got {mu.shape[1]}."
                    )
                bounds = [round(i * mu.shape[1] / n_factors) for i in range(n_factors + 1)]
            else:
                bounds = None
            losses, metrics = [], {}
            for d in range(n_factors):
                sub = mu if bounds is None else mu[:, bounds[d] : bounds[d + 1]]
                out = _proportional_rsa_loss(sub, mass[:, d : d + 1])
                if out is None:
                    continue
                loss_d, corr_d = out
                losses.append(loss_d)
                metrics[f"contrastive_rsa_corr_f{d}"] = corr_d.detach()
            if not losses:
                return mu.new_zeros(()), {}
            loss = torch.stack(losses).mean()
            metrics["contrastive"] = loss.detach()
            metrics["contrastive_rsa_corr"] = torch.stack(
                [metrics[f"contrastive_rsa_corr_f{d}"] for d in range(n_factors)
                 if f"contrastive_rsa_corr_f{d}" in metrics]
            ).mean()
            return weight * loss, metrics

        out = _proportional_rsa_loss(mu, mass)
        if out is None:                                           # batch is single-mass
            return mu.new_zeros(()), {}
        loss, corr = out
        metrics = {
            "contrastive": loss.detach(),
            "contrastive_rsa_corr": corr.detach(),
        }
        return weight * loss, metrics

    z = F.normalize(mu, dim=-1)
    sim = (z @ z.t()) / temperature  # (B, B) cosine / temperature
    eye = torch.eye(b, device=mu.device, dtype=torch.bool)
    neg_inf = torch.finfo(sim.dtype).min
    sim = sim.masked_fill(eye, neg_inf)  # exclude self from the partition function
    mass = mass.reshape(b, -1).to(mu.device)
    mass_dist = torch.cdist(mass, mass)
    positives = (mass_dist <= mass_delta) & (~eye)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # log-softmax over negatives+positives
    pos_count = positives.sum(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return mu.new_zeros(()), {}
    per_anchor = -(logp * positives).sum(1) / pos_count.clamp_min(1)
    loss = per_anchor[valid].mean()
    metrics = {
        "contrastive": loss.detach(),
        "contrastive_pos_per_anchor": pos_count.float().mean().detach(),
    }
    return weight * loss, metrics


def weighted_rollout_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    robot_dof: int,
    q_weight: float = 1.0,
    dq_weight: float = 0.2,
    object_pos_weight: float = 5.0,
    object_quat_weight: float = 0.5,
    object_lin_vel_weight: float = 1.0,
    object_ang_vel_weight: float = 0.5,
    channel_std: torch.Tensor | None = None,
    arm_dof: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``channel_std`` (state_dim,) standardizes every channel before its MSE.

    ``arm_dof`` restricts the q / dq terms to the first ``arm_dof`` joints. The
    gripper fingers carry no free coordinate in phase 3 -- they are closed on the
    payload and move under contact forces this model does not represent -- and they
    do not enter the object's forward kinematics, so their error is both unfixable
    and unscored. Standardized, their near-constant std (~1e-3 rad) would turn that
    unfixable residual into the dominant term.

    Without it the six weights act on raw units, and the units decide: measured on a
    trained run, the metre-scale object position carried 0.003% of the total while the
    quaternion and angular velocity carried 77% -- nominal 1:1 weights, effective
    1:3000. Dividing by the train-split per-channel std makes the nominal weights the
    real ones, for all six groups.
    """
    if channel_std is not None:
        pred = pred / channel_std
        target = target / channel_std
    offset = 2 * robot_dof
    n_q = robot_dof if arm_dof is None else arm_dof
    parts = {
        "q_mse": F.mse_loss(pred[..., :n_q], target[..., :n_q]),
        "dq_mse": F.mse_loss(pred[..., robot_dof : robot_dof + n_q], target[..., robot_dof : robot_dof + n_q]),
        "object_pos_mse": F.mse_loss(pred[..., offset : offset + 3], target[..., offset : offset + 3]),
        "object_quat_mse": F.mse_loss(pred[..., offset + 3 : offset + 7], target[..., offset + 3 : offset + 7]),
        "object_lin_vel_mse": F.mse_loss(pred[..., offset + 7 : offset + 10], target[..., offset + 7 : offset + 10]),
        "object_ang_vel_mse": F.mse_loss(pred[..., offset + 10 : offset + 13], target[..., offset + 10 : offset + 13]),
    }
    loss = (
        q_weight * parts["q_mse"]
        + dq_weight * parts["dq_mse"]
        + object_pos_weight * parts["object_pos_mse"]
        + object_quat_weight * parts["object_quat_mse"]
        + object_lin_vel_weight * parts["object_lin_vel_mse"]
        + object_ang_vel_weight * parts["object_ang_vel_mse"]
    )
    return loss, parts


def inverse_dynamics_loss(
    H: torch.Tensor,
    coriolis: torch.Tensor,
    gravity: torch.Tensor,
    qdd: torch.Tensor,
    tau: torch.Tensor,
    *,
    arm_dof: int | None = 7,
    tau_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""DeLaN's own objective: :math:`\|H\ddot q + c + g - \tau\|^2`, in N*m.

    This is not a different zero set from the rollout loss -- :math:`\ddot q =
    H^{-1}(\tau - c - g)` and :math:`H\ddot q + c + g = \tau` are one equation, and the
    rescaling :math:`(H, c, g) \to (\alpha H, \alpha c, (1-\alpha)\tau + \alpha g)` is
    a null direction of both. It is a different *metric on the residual*, and that is
    what matters. A network can only approximate :math:`\tau` from :math:`(q, \dot q)`
    (ceiling :math:`R^2 \approx 0.94` on the phase-3 arm joints; :math:`\tau` is a PD law,
    corr 0.998-0.999 against ``Kp(q_tgt - q) - Kd dq``), so the realisable member of the
    family carries :math:`(1-\alpha)(\tilde\tau - \tau)`. Torque space charges that
    linearly in :math:`\alpha`; acceleration space divides it by :math:`\alpha H` and
    saturates. On the real data :math:`\alpha: 1 \to 50` moves the :math:`\ddot q` error
    1183 -> 1161 rad/s^2 and the torque error 8.0 -> 398.9 N*m. The forward objective
    therefore lets the model buy insensitivity to its own misfit by inflating H -- which
    is what ``armpay_ctx`` did (H ~ 10x, c ~ 40x, g left fitting tau rather than gravity).

    ``arm_dof`` drops the two finger joints. They carry the grip's contact forces
    (rms |tau| 44 N*m against an |Hqdd + c + g| of 0.24) and this model has no contact
    term, so their residual is unfixable and would be the whole loss.

    ``tau_scale`` (arm_dof,) divides each joint's residual, so one weight can be read
    without joints 1 and 3 -- where the gravity lives -- swamping the rest.
    """
    n = H.shape[-1] if arm_dof is None else int(arm_dof)
    tau_hat = (H @ qdd.unsqueeze(-1)).squeeze(-1) + coriolis + gravity
    residual = (tau_hat - tau)[..., :n]
    if tau_scale is not None:
        residual = residual / tau_scale[:n]
    loss = (residual**2).mean()
    with torch.no_grad():
        metrics = {
            "id_rmse_nm": (tau_hat - tau)[..., :n].pow(2).mean().sqrt(),
            "id_tau_hat_rms": tau_hat[..., :n].pow(2).mean().sqrt(),
            "id_tau_rms": tau[..., :n].pow(2).mean().sqrt(),
            "id_inertial_rms": (H @ qdd.unsqueeze(-1)).squeeze(-1)[..., :n].pow(2).mean().sqrt(),
            "id_coriolis_rms": coriolis[..., :n].pow(2).mean().sqrt(),
            "id_gravity_rms": gravity[..., :n].pow(2).mean().sqrt(),
        }
    return loss, metrics


def mcg_supervision_loss(
    H: torch.Tensor,
    coriolis: torch.Tensor,
    gravity: torch.Tensor,
    H_gt: torch.Tensor,
    c_gt: torch.Tensor,
    g_gt: torch.Tensor,
    *,
    arm_dof: int = 7,
    normalize: bool = True,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    r"""Supervise the DeLaN decomposition term by term against ``GT_full``.

    The other two objectives in this study constrain only the SUM :math:`H\ddot q + c + g`
    (as :math:`\ddot q` or as :math:`\tau`), and the sum does not determine the split: the
    data never separates gravity from the PD controller's position term, because both are
    functions of :math:`q` alone and both are gradients of a scalar, so a learned potential
    may absorb the controller and stay inside DeLaN's structure. Measured on the rollout
    run: Franka joint 0 turns about the world vertical, where gravity can produce exactly
    zero torque, and the model puts 3.44 N*m there against a logged 0.0003.

    This term supplies what the data cannot. The target is ``GT_full = GT_arm + payload``
    from :mod:`cdelan.analysis.build_mcg_targets` -- NOT the logged arrays, which are
    arm-only and would train the model to behave as if there were no payload at all.

    ``normalize`` divides each term by the batch RMS of its own target, so the three
    weights are comparable: in raw units g is ~30 N*m, H ~1, c ~1, and an unnormalized sum
    would be gravity alone.
    """
    n = int(arm_dof)
    Hm, cm, gm = H[..., :n, :n], coriolis[..., :n], gravity[..., :n]
    Ht, ct, gt = H_gt[..., :n, :n], c_gt[..., :n], g_gt[..., :n]

    def term(pred, target):
        err = (pred - target).pow(2).mean()
        if normalize:
            err = err / target.pow(2).mean().clamp_min(1e-8)
        return err

    lH, lc, lg = term(Hm, Ht), term(cm, ct), term(gm, gt)
    wH, wc, wg = weights
    loss = wH * lH + wc * lc + wg * lg
    with torch.no_grad():
        metrics = {
            "mcg_H": lH,
            "mcg_c": lc,
            "mcg_g": lg,
            "mcg_H_rel_rmse": ((Hm - Ht).pow(2).mean() / Ht.pow(2).mean().clamp_min(1e-8)).sqrt(),
            "mcg_c_rel_rmse": ((cm - ct).pow(2).mean() / ct.pow(2).mean().clamp_min(1e-8)).sqrt(),
            "mcg_g_rel_rmse": ((gm - gt).pow(2).mean() / gt.pow(2).mean().clamp_min(1e-8)).sqrt(),
            # the free falsification test: gravity can produce no torque about joint 0
            "mcg_g_j0_rms": gm[..., 0].pow(2).mean().sqrt(),
            "mcg_g_j0_rms_gt": gt[..., 0].pow(2).mean().sqrt(),
        }
    return loss, metrics
