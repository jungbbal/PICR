"""Open-loop PICR rollout with a supplied context z (used to compare an adapted z against a stale one)."""
from __future__ import annotations

import torch

CUBE_SIDE = 0.048  # m, the lift cube


@torch.no_grad()
def roll_with_z(model, hs, ft, z) -> torch.Tensor:
    """`ArmPayloadWMDynamics.forward` with z supplied instead of encoded. -> (B, h, 7) object pos + quat."""
    r = model.robot_dof
    state = hs[:, -1]
    grasp = model.infer_grasp(state)
    q, dq = state[:, :r], state[:, r : 2 * r]
    out = []
    for k in range(ft.shape[1]):
        q, dq, _ = model._step_arm(q, dq, ft[:, k], z=z)
        out.append(model.assemble_state(q, dq, grasp)[:, 2 * r : 2 * r + 7])  # object pos + quat
    return torch.stack(out, 1)
