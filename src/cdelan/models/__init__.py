"""The one structure this package carries: a 9-DOF DeLaN arm with an FK-carried payload.

``arm_payload`` puts DeLaN over the arm's own coordinates only and derives the object
pose from forward kinematics plus a grasp offset measured once from the history. The
15-DOF variants, the contact-coupled variants and the vision models that used to live
here are gone -- see ``docs/PROVENANCE.md`` for what was dropped and why.
"""

from .arm_payload_dynamics import (
    ArmPayloadWMDynamics,
    ArmPayloadWMDynamicsConfig,
    QddResidual,
    build_arm_payload_wm_dynamics,
)
from .context import ContextEncoder, ContextOutput, kl_divergence_standard_normal, reparameterize
from .delan import DeLaNCore, DeLaNRobotDynamics, DelanTerms, RobotDynamicsOutput

__all__ = [
    "ArmPayloadWMDynamics",
    "ArmPayloadWMDynamicsConfig",
    "QddResidual",
    "ContextEncoder",
    "ContextOutput",
    "DeLaNCore",
    "DeLaNRobotDynamics",
    "DelanTerms",
    "RobotDynamicsOutput",
    "build_arm_payload_wm_dynamics",
    "kl_divergence_standard_normal",
    "reparameterize",
]
