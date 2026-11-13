from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp, safe_solve

ODE_SOLVERS = ("euler", "trapezoid", "rk4", "trapezoid_pc", "trapezoid_mid")


def validate_ode_solver(value: str) -> str:
    solver = value.lower()
    if solver not in ODE_SOLVERS:
        joined = ", ".join(ODE_SOLVERS)
        raise ValueError(f"ode_solver must be one of ({joined}); got {value!r}.")
    return solver


def safe_cholesky_solve(
    mass_matrix: torch.Tensor,
    rhs: torch.Tensor,
    damping: float = 1.0e-4,
) -> torch.Tensor:
    return safe_solve(mass_matrix, rhs, damping=damping)


@dataclass
class DelanTerms:
    H: torch.Tensor
    L: torch.Tensor
    dL_dq: torch.Tensor
    l_diag: torch.Tensor
    l_offdiag: torch.Tensor
    g: torch.Tensor
    coriolis: torch.Tensor
    # H^-1, and only when the core parameterizes the INVERSE inertia -- see
    # `DeLaNCore.parameterize_inverse`. None means "form ddq by solving with H", the path every
    # checkpoint before took. Deliberately NOT in `as_aux`: every consumer of that
    # dict (the mcg losses, mcg_vs_gt, the analysis scripts) is keyed on the physical terms, and
    # W is a statement about how the acceleration is formed, not a new physical quantity.
    W: torch.Tensor | None = None
    # dH/dq (B, n, n, n), set by cores whose H is not a single Cholesky factor (BasisDeLaNCore), so the
    # Christoffel term can be re-contracted at another velocity without a second network pass.
    dH_dq: torch.Tensor | None = None
    # The learned acceleration residual, when ArmPayloadWMDynamics.qdd_residual is on. Set by
    # `acceleration`, not by the core -- it is not a DeLaN term, it is what sits beside them.
    qdd_residual: torch.Tensor | None = None

    def as_aux(self) -> dict[str, torch.Tensor]:
        aux = {
            "H": self.H,
            "L": self.L,
            "dL_dq": self.dL_dq,
            "l_diag": self.l_diag,
            "l_offdiag": self.l_offdiag,
            "g": self.g,
            "coriolis": self.coriolis,
        }
        if self.qdd_residual is not None:
            aux["qdd_residual"] = self.qdd_residual
        return aux


@dataclass
class RobotDynamicsOutput:
    next_q: torch.Tensor
    next_dq: torch.Tensor
    ddq: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class DerivativeMLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_dim), nn.SiLU()]
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        super().__init__(*layers)

    @staticmethod
    def silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    def forward_with_jacobian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_dim = x.shape
        h = x
        jac = torch.eye(in_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, in_dim, in_dim)
        for layer in self:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                jac = torch.einsum("oi,bik->bok", layer.weight, jac)
            elif isinstance(layer, nn.SiLU):
                deriv = self.silu_derivative(h)
                h = F.silu(h)
                jac = deriv.unsqueeze(-1) * jac
            else:
                raise TypeError(f"Unsupported derivative layer: {type(layer).__name__}")
        return h, jac


class FiLMLayer(nn.Module):
    """Linear layer modulated by FiLM parameters from latent context z."""

    def __init__(self, hidden_dim: int, latent_dim: int, modulation_limit: float = 0.1,
                 temperature: float = 1.0, beta_modulation_limit: float | None = None,
                 beta_temperature: float | None = None) -> None:
        super().__init__()
        self.modulation_limit = float(modulation_limit)
        # Optional separate gate for the shift (beta) channel; None = same as the scale (gamma) gate.
        # Lets the 2x2 (cap x roll-off) control be applied to gamma only .
        self.beta_modulation_limit = float(beta_modulation_limit) if beta_modulation_limit is not None else None
        self.beta_temperature = float(beta_temperature) if beta_temperature is not None else None
        # Gate = limit * T * tanh(x / T). T = 1 is the original gate. T > 1 keeps the slope at the origin
        # (limit) but pushes the tanh roll-off out by T: at |x| = 1 the slope is limit * tanh'(1/T), i.e.
        # 0.42 * limit for T = 1 and 0.79 * limit for T = 2, and the cap becomes limit * T. Motivated by the
        # measurement that the units carrying the payload-mass change sit at |x| 0.5-1.0 for a
        # 3 kg payload, where tanh' is 0.5-0.6 (logs/qddlabel/film_ranges.py).
        self.temperature = float(temperature)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film_gamma = nn.Linear(latent_dim, hidden_dim)
        self.film_beta = nn.Linear(latent_dim, hidden_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def modulation(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(scale, shift) from z. With temperature 1 this is the original ``1 + limit * tanh`` bit for bit."""
        lim_b = self.modulation_limit if self.beta_modulation_limit is None else self.beta_modulation_limit
        t_b = self.temperature if self.beta_temperature is None else self.beta_temperature
        if self.temperature == 1.0:
            scale = 1.0 + self.modulation_limit * torch.tanh(self.film_gamma(z))
        else:
            t = self.temperature
            scale = 1.0 + self.modulation_limit * t * torch.tanh(self.film_gamma(z) / t)
        if t_b == 1.0:
            shift = lim_b * torch.tanh(self.film_beta(z))
        else:
            shift = lim_b * t_b * torch.tanh(self.film_beta(z) / t_b)
        return scale, shift

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        scale, shift = self.modulation(z)
        return F.silu(scale * h + shift)


class FiLMDerivativeMLP(nn.Module):
    """DerivativeMLP replacement with FiLM blocks conditioned on latent z.

    The Jacobian is with respect to x only. z is treated as a fixed per-window
    latent, matching the DeLaN requirement for dL/dq.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        latent_dim: int,
        modulation_limit: float = 0.1,
        temperature: float = 1.0,
        beta_modulation_limit: float | None = None,
        beta_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.trunk = nn.Linear(in_dim, hidden_dim)
        self.film_layers = nn.ModuleList(
            [FiLMLayer(hidden_dim, latent_dim, modulation_limit=modulation_limit, temperature=temperature,
                       beta_modulation_limit=beta_modulation_limit, beta_temperature=beta_temperature)
             for _ in range(depth)]
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def silu_derivative(x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.trunk(x))
        for layer in self.film_layers:
            h = layer(h, z)
        return self.out(h)

    def forward_with_jacobian(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, in_dim = x.shape
        pre = self.trunk(x)
        deriv = self.silu_derivative(pre)
        h = F.silu(pre)
        jac = torch.eye(in_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, in_dim, in_dim)
        jac = torch.einsum("oi,bik->bok", self.trunk.weight, jac)
        jac = deriv.unsqueeze(-1) * jac

        for layer in self.film_layers:
            linear_h = layer.linear(h)
            scale, shift = layer.modulation(z)
            pre = scale * linear_h + shift
            h = F.silu(pre)

            jac = torch.einsum("oi,bik->bok", layer.linear.weight, jac)
            jac = scale.unsqueeze(-1) * jac
            jac = self.silu_derivative(pre).unsqueeze(-1) * jac

        out = self.out(h)
        jac = torch.einsum("oi,bik->bok", self.out.weight, jac)
        return out, jac


class DeLaNCore(nn.Module):
    """DeLaN robot core producing H(q), c(q,dq), and g(q)."""

    def __init__(
        self,
        robot_dof: int,
        hidden_dim: int,
        eps: float = 1.0e-4,
        diag_eps: float = 1.0e-3,
        armature_eps: float = 0.0,
        latent_dim: int | None = None,
        use_film: bool = False,
        film_depth: int = 2,
        film_modulation_limit: float = 0.1,
        parameterize_inverse: bool = False,
        film_temperature: float = 1.0,
        film_beta_modulation_limit: float | None = None,
        film_beta_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.eps = eps
        self.diag_eps = diag_eps
        # `armature_eps` is TrainConfig.delan_eps_diag. Three eps-like names now live in this
        # constructor and they are three different things:
        #   eps          1e-4  numerical guard, keeps L L' strictly positive definite
        #   diag_eps     1e-3  floor under the softplus on L's own diagonal
        #   armature_eps  cfg  the ROTOR INERTIA the recording does not have
        # The last one is a modelling choice, not a numerical one: PhysX logged
        # `robot_joint_params/joint_armature` as exactly 0, so the true wrist inertia is
        # 0.0036 kg m^2 and `H^-1` amplifies a torque error there by ~278 rad/s^2 per N*m.
        # Declaring it here puts it in H everywhere H is used -- the inverse-dynamics loss sees
        # (H + armature_eps*I) qdd, the rollout inverts (H + armature_eps*I), and `mcg_vs_gt`
        # reads the same H -- which is the whole point: the two objectives are then asked about
        # the SAME operator. It is a fixed constant, never a parameter; there is nothing to
        # give `requires_grad` to.
        self.armature_eps = float(armature_eps)
        # Let the net's Cholesky factor build W = H^-1 instead of H.
        #
        # The rollout only ever needs `ddq = H^-1 (tau - c - g)`, and today it gets there by
        # solving with H. That makes the gradient the objective sends back to L pass through the
        # inverse TWICE -- d(H^-1 r)/dL = -H^-1 (dH/dL) H^-1 r -- so a direction where H is soft
        # has its gradient amplified by cond(H)^2, and the cheapest way for the optimiser to calm
        # that down is to flatten H's spectrum. It duly does: the learned H comes out at cond 31
        # where the logged GT is 1060 (see docs; `mcg_vs_gt`). The model is not choosing an
        # unphysical inertia because the data asks for one, it is choosing it because of how the
        # operator is parameterized.
        #
        # Factoring the inverse removes exactly that: `ddq = W r` is a matmul, dW/dL carries no
        # inverse at all, and the amplification drops from cond^2 to cond. H is then a DERIVED
        # quantity (`cholesky_inverse(W)`), which is the honest ordering -- the rollout is what
        # the loss measures, so the rollout's own operator should be the one with parameters.
        #
        # The Christoffel term still needs dH/dq, and that identity costs one inverse per call:
        # dH/dq = -H (dW/dq) H. That inverse is not in the acceleration's gradient path to the
        # same degree, and c is O(dq^2) against tau and g, so it is the cheap half to pay for.
        self.parameterize_inverse = bool(parameterize_inverse)
        n_offdiag = robot_dof * (robot_dof - 1) // 2
        self.use_film = bool(use_film) and latent_dim is not None and latent_dim > 0
        self.latent_dim = int(latent_dim) if self.use_film else None
        self.film_modulation_limit = float(film_modulation_limit)
        self.film_temperature = float(film_temperature)
        self._film_beta = dict(beta_modulation_limit=film_beta_modulation_limit, beta_temperature=film_beta_temperature)

        if self.use_film:
            self.v_net = FiLMDerivativeMLP(
                robot_dof, hidden_dim, 1, depth=film_depth, latent_dim=int(latent_dim),
                modulation_limit=self.film_modulation_limit, temperature=self.film_temperature, **self._film_beta,
            )
            self.l_diag_net = FiLMDerivativeMLP(
                robot_dof,
                hidden_dim,
                robot_dof,
                depth=film_depth,
                latent_dim=int(latent_dim),
                modulation_limit=self.film_modulation_limit,
                temperature=self.film_temperature, **self._film_beta,
            )
            self.l_offdiag_net = FiLMDerivativeMLP(
                robot_dof,
                hidden_dim,
                n_offdiag,
                depth=film_depth,
                latent_dim=int(latent_dim),
                modulation_limit=self.film_modulation_limit,
                temperature=self.film_temperature, **self._film_beta,
            )
        else:
            self.v_net = DerivativeMLP(robot_dof, hidden_dim, 1, depth=2)
            self.l_diag_net = DerivativeMLP(robot_dof, hidden_dim, robot_dof, depth=2)
            self.l_offdiag_net = DerivativeMLP(robot_dof, hidden_dim, n_offdiag, depth=2)

        offdiag = torch.tril_indices(row=robot_dof, col=robot_dof, offset=-1)
        self.register_buffer("offdiag_row", offdiag[0], persistent=False)
        self.register_buffer("offdiag_col", offdiag[1], persistent=False)
        self.register_buffer("diag_idx", torch.arange(robot_dof), persistent=False)

    def make_l_and_derivatives(self, q: torch.Tensor, z: torch.Tensor | None = None):
        batch = q.shape[0]
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNCore.use_film=True requires latent z.")
            l_diag_raw, dl_diag_raw_dq = self.l_diag_net.forward_with_jacobian(q, z)
            l_offdiag, dl_offdiag_dq = self.l_offdiag_net.forward_with_jacobian(q, z)
        else:
            l_diag_raw, dl_diag_raw_dq = self.l_diag_net.forward_with_jacobian(q)
            l_offdiag, dl_offdiag_dq = self.l_offdiag_net.forward_with_jacobian(q)

        L = q.new_zeros((batch, self.robot_dof, self.robot_dof))
        dL_dq = q.new_zeros((batch, self.robot_dof, self.robot_dof, self.robot_dof))
        L[:, self.offdiag_row, self.offdiag_col] = l_offdiag
        dL_dq[:, self.offdiag_row, self.offdiag_col, :] = dl_offdiag_dq

        l_diag = F.softplus(l_diag_raw) + self.diag_eps
        dl_diag_dq = torch.sigmoid(l_diag_raw).unsqueeze(-1) * dl_diag_raw_dq
        L[:, self.diag_idx, self.diag_idx] = l_diag
        dL_dq[:, self.diag_idx, self.diag_idx, :] = dl_diag_dq
        return L, dL_dq, l_diag, l_offdiag

    def inertia(self, q: torch.Tensor, z: torch.Tensor | None = None):
        """Returns ``(H, W, H_raw, L, dL_dq, l_diag, l_offdiag)``.

        ``W`` is ``H^-1`` and ``H_raw`` is ``H`` before the rotor inertia, both ``None`` on the
        historical path. ``H_raw`` is what the ``dH/dq = -H (dW/dq) H`` identity needs: the
        armature is a constant, so it drops out of the derivative, and using ``H`` there instead
        would silently fold it in twice.
        """
        L, dL_dq, l_diag, l_offdiag = self.make_l_and_derivatives(q, z=z)
        eye = torch.eye(self.robot_dof, device=q.device, dtype=q.dtype).unsqueeze(0)
        if not self.parameterize_inverse:
            # Left character-for-character as it was, so every pre-existing checkpoint replays
            # bit-identically rather than one ulp off a re-associated sum.
            H = L @ L.transpose(-1, -2) + (self.eps + self.armature_eps) * eye
            return H, None, None, L, dL_dq, l_diag, l_offdiag

        W_raw = L @ L.transpose(-1, -2) + self.eps * eye
        H_raw = torch.cholesky_inverse(torch.linalg.cholesky(W_raw))
        if self.armature_eps:
            # The rotor inertia is defined ON H, so it has to be added there and the inverse
            # retaken; adding it to W would be a different (and wrong) operator.
            H = H_raw + self.armature_eps * eye
            W = torch.cholesky_inverse(torch.linalg.cholesky(H))
        else:
            H, W = H_raw, W_raw
        return H, W, H_raw, L, dL_dq, l_diag, l_offdiag

    @staticmethod
    def _d_gram_dq(L: torch.Tensor, dL_dq: torch.Tensor) -> torch.Tensor:
        """``d(L L')/dq_k`` -- the bilinear form ``coriolis_centrifugal`` builds inline."""
        return (
            torch.einsum("bim,bjmk->bijk", L, dL_dq)
            + torch.einsum("bimk,bjm->bijk", dL_dq, L)
        )

    @staticmethod
    def _coriolis_from_dH(dH_dq: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        """The same Christoffel term as ``coriolis_centrifugal``, from ``dH/dq`` directly.

        ``coriolis_centrifugal`` reaches it through L because on that path H is ``L L'`` and the
        factor is what the net holds. Here the net holds W, so dH/dq arrives already assembled
        and only the contraction is left.
        """
        dH_dt = torch.einsum("bijk,bk->bij", dH_dq, dq)
        dH_dt_dq = torch.einsum("bij,bj->bi", dH_dt, dq)
        kinetic_grad = torch.einsum("bi,bijk,bj->bk", dq, dH_dq, dq)
        return dH_dt_dq - 0.5 * kinetic_grad

    @staticmethod
    def coriolis_centrifugal(L: torch.Tensor, dL_dq: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
        dL_dt = torch.einsum("bijk,bk->bij", dL_dq, dq)
        dH_dt = L @ dL_dt.transpose(-1, -2) + dL_dt @ L.transpose(-1, -2)
        dH_dt_dq = torch.einsum("bij,bj->bi", dH_dt, dq)
        dH_dq = torch.einsum("bim,bjmk->bijk", L, dL_dq) + torch.einsum("bimk,bjm->bijk", dL_dq, L)
        kinetic_grad = torch.einsum("bi,bijk,bj->bk", dq, dH_dq, dq)
        return dH_dt_dq - 0.5 * kinetic_grad

    def forward(self, q: torch.Tensor, dq: torch.Tensor, z: torch.Tensor | None = None) -> DelanTerms:
        H, W, H_raw, L, dL_dq, l_diag, l_offdiag = self.inertia(q, z=z)
        if self.use_film:
            if z is None:
                raise ValueError("DeLaNCore.use_film=True requires latent z.")
            V, dV_dq = self.v_net.forward_with_jacobian(q, z)
        else:
            V, dV_dq = self.v_net.forward_with_jacobian(q)
        if self.parameterize_inverse:
            dH_dq = -torch.einsum("bim,bmnk,bnj->bijk", H_raw, self._d_gram_dq(L, dL_dq), H_raw)
            coriolis = self._coriolis_from_dH(dH_dq, dq)
        else:
            coriolis = self.coriolis_centrifugal(L, dL_dq, dq)
        g= dV_dq.squeeze(1)
        return DelanTerms(H=H, L=L, dL_dq=dL_dq, l_diag=l_diag, l_offdiag=l_offdiag, g=g,
                          coriolis=coriolis, W=W)



class BasisDeLaNCore(nn.Module):
    """Lagrangian that is LINEAR in a non-negative context weight vector .

        H(q, z) = sum_k w_k(z) L_k(q) L_k(q)' + eps I,   V(q, z) = sum_k w_k(z) V_k(q),   w_k >= 0

    WHY. Kinetic and potential energy are linear in every body's mass: scaling link i by s_i scales
    its Gram contribution to H and its share of V by the same s_i, and a rigidly grasped payload is
    one more such body. Both recordings randomise all arm link masses by +-10% per episode on top of
    the payload, so the true context is ~9 mass-like numbers that enter exactly this way. The FiLM
    core has to emulate that product with tanh-limited modulations of hidden units; z then carried
    the payload mass and none of the arm (LOO R2 <= 0.02 on the 8 link scales), which left a
    per-episode gravity bias of ~1 N m on j2/j4 and the rest-window drift it causes.

    The K bodies are learned, not the URDF's links: each has its own Cholesky factor field and its
    own potential, sharing the three trunks. The SAME w multiplies a body's H_k and V_k -- that tie
    is the physics (one mass per body) and it is what lets gravity, identifiable at rest, inform the
    inertia. H stays symmetric positive definite (non-negative sum of Grams), the Christoffel
    operator is linear in H so c is the w-weighted sum too, and g is the gradient of a potential.
    z enters ONLY through w.
    """

    def __init__(
        self,
        robot_dof: int,
        hidden_dim: int,
        *,
        latent_dim: int,
        n_basis: int = 12,
        depth: int = 3,
        eps: float = 1.0e-4,
        diag_eps: float = 1.0e-3,
        armature_eps: float = 0.0,
        identity_head: bool = False,
    ) -> None:
        super().__init__()
        self.robot_dof = robot_dof
        self.n_basis = int(n_basis)
        # z IS the weights' pre-activation, w = softplus(z + b): needed when the weights are solved
        # for (least squares on the history) and have to be handed back as a z every caller of
        # core(q, dq, z) understands. Requires latent_dim == n_basis.
        self.identity_head = bool(identity_head)
        if self.identity_head and int(latent_dim) != int(n_basis):
            raise ValueError("identity_head needs latent_dim == n_basis.")
        self.eps = eps
        self.diag_eps = diag_eps
        self.armature_eps = float(armature_eps)
        self.latent_dim = int(latent_dim)
        n_offdiag = robot_dof * (robot_dof - 1) // 2
        self.n_offdiag = n_offdiag
        k = self.n_basis
        self.v_net = DerivativeMLP(robot_dof, hidden_dim, k, depth=depth)
        self.l_diag_net = DerivativeMLP(robot_dof, hidden_dim, k * robot_dof, depth=depth)
        self.l_offdiag_net = DerivativeMLP(robot_dof, hidden_dim, k * n_offdiag, depth=depth)
        # w = softplus(A z + b). A starts at zero (z is inert at step 0, as with FiLM) and b puts
        # every weight at 1/K, so the summed H starts at the scale of a single DeLaN core.
        self.weight_head = nn.Linear(self.latent_dim, k)
        nn.init.zeros_(self.weight_head.weight)
        w0 = torch.tensor(1.0 / k)
        nn.init.constant_(self.weight_head.bias, float(torch.log(torch.expm1(w0))))
        if self.identity_head:
            with torch.no_grad():
                self.weight_head.weight.copy_(torch.eye(k))
            self.weight_head.weight.requires_grad_(False)

        offdiag = torch.tril_indices(row=robot_dof, col=robot_dof, offset=-1)
        self.register_buffer("offdiag_row", offdiag[0], persistent=False)
        self.register_buffer("offdiag_col", offdiag[1], persistent=False)
        self.register_buffer("diag_idx", torch.arange(robot_dof), persistent=False)

    def weights(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.weight_head(z))

    def weights_to_z(self, w: torch.Tensor) -> torch.Tensor:
        """Inverse of ``weights`` under ``identity_head``: the z whose weights are ``w`` (> 0)."""
        return torch.log(torch.expm1(w)) - self.weight_head.bias

    def basis_terms(self, q: torch.Tensor):
        """Per-body ``(H_k, dH_k/dq, dV_k/dq)``: (B, K, n, n), (B, K, n, n, n), (B, K, n)."""
        b, n, k = q.shape[0], self.robot_dof, self.n_basis
        ld_raw, dld_raw = self.l_diag_net.forward_with_jacobian(q)
        lo, dlo = self.l_offdiag_net.forward_with_jacobian(q)
        ld_raw = ld_raw.reshape(b, k, n)
        L = q.new_zeros((b, k, n, n))
        dL = q.new_zeros((b, k, n, n, n))
        L[:, :, self.offdiag_row, self.offdiag_col] = lo.reshape(b, k, self.n_offdiag)
        dL[:, :, self.offdiag_row, self.offdiag_col, :] = dlo.reshape(b, k, self.n_offdiag, n)
        L[:, :, self.diag_idx, self.diag_idx] = F.softplus(ld_raw) + self.diag_eps
        dL[:, :, self.diag_idx, self.diag_idx, :] = (
            torch.sigmoid(ld_raw).unsqueeze(-1) * dld_raw.reshape(b, k, n, n)
        )
        Hk = L @ L.transpose(-1, -2)
        dHk = torch.einsum("bkim,bkjmc->bkijc", L, dL) + torch.einsum("bkimc,bkjm->bkijc", dL, L)
        _, dV = self.v_net.forward_with_jacobian(q)
        return Hk, dHk, dV, L

    def forward(self, q: torch.Tensor, dq: torch.Tensor, z: torch.Tensor | None = None) -> DelanTerms:
        if z is None:
            raise ValueError("BasisDeLaNCore requires latent z: the body weights are its only context path.")
        w = self.weights(z)
        Hk, dHk, dV, L = self.basis_terms(q)
        eye = torch.eye(self.robot_dof, device=q.device, dtype=q.dtype).unsqueeze(0)
        H = torch.einsum("bk,bkij->bij", w, Hk) + (self.eps + self.armature_eps) * eye
        dH_dq = torch.einsum("bk,bkijc->bijc", w, dHk)
        coriolis = DeLaNCore._coriolis_from_dH(dH_dq, dq)
        g = torch.einsum("bk,bkn->bn", w, dV)
        empty = H.new_zeros((H.shape[0], 0))
        # L / dL_dq of the SUM are not the sum of the factors; no caller outside the FiLM core
        # reads them, so hand back the per-body factors' weighted diagonal as a diagnostic only.
        return DelanTerms(H=H, L=H.new_zeros(H.shape), dL_dq=dH_dq.new_zeros(dH_dq.shape),
                          l_diag=torch.diagonal(H, dim1=-2, dim2=-1), l_offdiag=empty, g=g,
                          coriolis=coriolis, dH_dq=dH_dq)

    def regressor(self, q: torch.Tensor, dq: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
        """``Y`` with ``tau = Y w + (eps + armature) qdd``: (B, n, K), column k = H_k qdd + c_k + g_k."""
        Hk, dHk, dV, _ = self.basis_terms(q)
        b, k = Hk.shape[:2]
        ck = DeLaNCore._coriolis_from_dH(
            dHk.reshape(b * k, *dHk.shape[2:]), dq.repeat_interleave(k, dim=0)
        ).reshape(b, k, -1)
        y = torch.einsum("bkij,bj->bki", Hk, qdd) + ck + dV
        return y.transpose(1, 2)


class ResidualDeLaNCore(nn.Module):
    """``H = H_arm(q) + dH(q, z)``, and the same split for c and g.

    The arm's own M, C, G are not a mystery -- Isaac logs them every step, and arm-alone already
    reproduces GT_full at rel_rmse 0.212 / 0.287 / 0.174 (H / c / g). A single core has to
    rediscover all of that from the rollout, gets it ~1.2x too large on every joint, and that
    error is 4-6x the payload signal it is supposed to expose. Here the arm branch is fitted
    once against the logged arrays (a clean supervised regression -- no identifiability problem)
    and frozen, and only the payload branch sees z.

    The split is structurally exact, not an approximation:

      * the Christoffel operator is linear in H, so ``coriolis(H_arm) + coriolis(H_pay)`` IS
        ``coriolis(H_arm + H_pay)`` -- see ``DeLaNCore.coriolis_centrifugal``, which is bilinear
        in (L, dL_dq) and therefore linear in the H each branch builds;
      * gravity is the gradient of a potential, and gradients add: ``g_arm + g_pay = grad(V_arm + V_pay)``;
      * both branches build ``H = L L' + eps I``, positive definite, so the sum is too -- and a
        payload's contribution to the mass matrix really is PSD (``m Jc'Jc + Jw' I Jw`` is a sum
        of Gram matrices), which the single-core form had no way to express.

    ``L``/``dL_dq``/``l_diag``/``l_offdiag`` in the returned terms are the PAYLOAD branch's. No
    caller outside this module reads them; the sum's Cholesky factor is not L_arm + L_pay and
    must not be reconstructed that way.

    Note the payload branch needs a FiLM modulation limit near 1.0, not the 0.1 the single-core
    runs use. There z only had to *tilt* a mostly-right H; here its branch is the whole payload
    term, which has to vanish for a 0.1 kg object and be large for a 2 kg one.
    """

    def __init__(
        self,
        robot_dof: int,
        hidden_dim: int,
        *,
        latent_dim: int,
        eps: float = 1.0e-4,
        diag_eps: float = 1.0e-3,
        armature_eps: float = 0.0,
        film_depth: int = 2,
        film_modulation_limit: float = 1.0,
        arm_hidden_dim: int | None = None,
        parameterize_inverse: bool = False,
        film_temperature: float = 1.0,
        film_beta_modulation_limit: float | None = None,
        film_beta_temperature: float | None = None,
    ) -> None:
        super().__init__()
        if parameterize_inverse:
            # The split is exact because H, c and g are ADDITIVE across the two branches. The
            # inverse is not: (H_arm + H_pay)^-1 is not W_arm + W_pay, so there is no branch
            # whose factor could be the sum's W and the acceleration would have to invert the
            # sum anyway -- which is the one thing the parameterization exists to avoid. Pick
            # one: either the arm prior or the factored inverse.
            raise ValueError(
                "ResidualDeLaNCore does not support parameterize_inverse: the branches add in "
                "H, and the inverse of a sum is not the sum of the inverses. Run arm_prior "
                "against the solve path, or the inverse parameterization with a single core."
            )
        self.robot_dof = robot_dof
        # The armature goes on the ARM branch alone. `forward` returns H_arm + H_pay, so giving
        # it to both branches would put 2*eps into the sum; and physically the rotor inertia
        # belongs to the arm, not to whatever it is holding.
        self.arm = DeLaNCore(
            robot_dof, arm_hidden_dim or hidden_dim, eps=eps, diag_eps=diag_eps,
            armature_eps=armature_eps, use_film=False
        )
        self.payload = DeLaNCore(
            robot_dof, hidden_dim, eps=eps, diag_eps=diag_eps, latent_dim=latent_dim,
            use_film=True, film_depth=film_depth, film_modulation_limit=film_modulation_limit,
            film_temperature=film_temperature, film_beta_modulation_limit=film_beta_modulation_limit,
            film_beta_temperature=film_beta_temperature,
        )

    def freeze_arm(self) -> None:
        """Hold the fitted arm branch fixed. Without this the payload's error can be paid for
        by drifting the arm, which is exactly the confusion the split exists to remove."""
        for prm in self.arm.parameters():
            prm.requires_grad_(False)
        self.arm.eval()

    def train(self, mode: bool = True):  # noqa: D102 - keep a frozen arm in eval mode
        super().train(mode)
        if not any(prm.requires_grad for prm in self.arm.parameters()):
            self.arm.eval()
        return self

    def forward(self, q: torch.Tensor, dq: torch.Tensor, z: torch.Tensor | None = None) -> DelanTerms:
        if z is None:
            raise ValueError("ResidualDeLaNCore requires latent z: the payload branch is FiLM-conditioned.")
        arm = self.arm(q, dq)
        pay = self.payload(q, dq, z=z)
        return DelanTerms(
            H=arm.H + pay.H,
            L=pay.L,
            dL_dq=pay.dL_dq,
            l_diag=pay.l_diag,
            l_offdiag=pay.l_offdiag,
            g=arm.g + pay.g,
            coriolis=arm.coriolis + pay.coriolis,
        )


class DeLaNRobotDynamics(nn.Module):
    """Robot dynamics with no residual, damping/friction, or context branch."""

    def __init__(
        self,
        robot_dof: int = 9,
        torque_dim: int = 9,
        hidden_dim: int = 256,
        dt: float = 0.02,
        latent_dim: int | None = None,
        use_film: bool = False,
        film_depth: int = 2,
        film_modulation_limit: float = 0.1,
        ode_solver: str = "euler",
        parameterize_inverse: bool = False,
    ) -> None:
        super().__init__()
        if torque_dim != robot_dof:
            raise ValueError("DeLaNRobotDynamics expects torque_dim == robot_dof.")
        self.robot_dof = robot_dof
        self.torque_dim = torque_dim
        self.dt = dt
        self.ode_solver = validate_ode_solver(ode_solver)
        self.use_film = bool(use_film) and latent_dim is not None and latent_dim > 0
        self.core = DeLaNCore(
            robot_dof=robot_dof,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim if self.use_film else None,
            use_film=self.use_film,
            film_depth=film_depth,
            film_modulation_limit=film_modulation_limit,
            parameterize_inverse=parameterize_inverse,
        )

    def forward(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        torque: torch.Tensor,
        external_torque: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> RobotDynamicsOutput:
        external = torch.zeros_like(torque) if external_torque is None else external_torque
        effective_torque = torque + external
        ddq, terms = self.acceleration(q, dq, effective_torque, z=z)
        inertial = (terms.H @ ddq.unsqueeze(-1)).squeeze(-1)
        next_q, next_dq = self.integrate(q, dq, effective_torque, ddq=ddq, z=z)

        aux = terms.as_aux()
        aux.update(
            {
                "torque": torque,
                "external_torque": external,
                "effective_torque": effective_torque,
                "ddq": ddq,
                "inertial": inertial,
                "inverse_dynamics_tau": inertial + terms.coriolis + terms.g,
                "ode_solver": torch.tensor(0 if self.ode_solver == "euler" else 1, device=q.device),
            }
        )
        return RobotDynamicsOutput(next_q=next_q, next_dq=next_dq, ddq=ddq, aux=aux)

    def acceleration(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DelanTerms]:
        terms = self.core(q, dq, z=z if self.use_film else None)
        rhs = effective_torque - terms.g - terms.coriolis
        # The guard and the clamp stay on BOTH paths on purpose: the point of the comparison is
        # the operator, so anything downstream of it has to be identical or the ablation is
        # reading two changes at once. On the W path the clamp should simply never bind.
        ddq = torch.einsum("bij,bj->bi", terms.W, rhs) if terms.W is not None \
            else safe_solve(terms.H, rhs)
        ddq = torch.nan_to_num(ddq, nan=0.0, posinf=1e3, neginf=-1e3)
        ddq = ddq.clamp(-1e3, 1e3)
        return ddq, terms

    def state_derivative(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ddq, _terms = self.acceleration(q, dq, effective_torque, z=z)
        return dq, ddq

    def integrate(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        ddq: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ode_solver in ("euler", "trapezoid"):
            next_dq = dq + ddq * self.dt
            mean_dq = 0.5 * (dq + next_dq) if self.ode_solver == "trapezoid" else next_dq
            next_q = q + mean_dq * self.dt
            return next_q, next_dq
        return self._rk4_integrate(q, dq, effective_torque, z=z)

    def _rk4_integrate(
        self,
        q: torch.Tensor,
        dq: torch.Tensor,
        effective_torque: torch.Tensor,
        *,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.dt
        k1_q, k1_dq = self.state_derivative(q, dq, effective_torque, z=z)
        k2_q, k2_dq = self.state_derivative(
            q + 0.5 * dt * k1_q,
            dq + 0.5 * dt * k1_dq,
            effective_torque,
            z=z,
        )
        k3_q, k3_dq = self.state_derivative(
            q + 0.5 * dt * k2_q,
            dq + 0.5 * dt * k2_dq,
            effective_torque,
            z=z,
        )
        k4_q, k4_dq = self.state_derivative(
            q + dt * k3_q,
            dq + dt * k3_dq,
            effective_torque,
            z=z,
        )
        next_q = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        next_dq = dq + (dt / 6.0) * (k1_dq + 2.0 * k2_dq + 2.0 * k3_dq + k4_dq)
        return next_q, next_dq


DeLaNStructuredTerms = DeLaNCore
