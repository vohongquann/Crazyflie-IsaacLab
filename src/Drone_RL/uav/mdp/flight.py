"""Constants and geometry shared by the layers, the observation terms and the rewards (pure torch)."""
from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from Drone_RL.uav import uav_cfg as U

GRAVITY = 9.81
WEIGHT_N = U.DRONE_MASS_TOTAL_KG * GRAVITY


def thrust_axis(quat: torch.Tensor) -> torch.Tensor:
    """Body z axis in the world frame: where the thrust points (N, 3)."""
    z = torch.zeros_like(quat[:, :3])
    z[:, 2] = 1.0
    return quat_apply(quat, z)


def tilt_error(quat: torch.Tensor, wanted_acceleration: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between the body up axis and the thrust direction that gives ``wanted_acceleration`` (world)."""
    force_per_mass = wanted_acceleration + torch.tensor([0.0, 0.0, GRAVITY], device=wanted_acceleration.device)
    wanted_up = force_per_mass / force_per_mass.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.acos((thrust_axis(quat) * wanted_up).sum(-1).clamp(-1.0, 1.0))
