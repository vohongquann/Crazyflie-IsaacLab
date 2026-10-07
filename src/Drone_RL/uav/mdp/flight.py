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


def up_axis_from_angles(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Body z axis in the world frame of the orientation Rz(yaw) Ry(pitch) Rx(roll) (N, 3): where the thrust points."""
    ux, uy = torch.sin(pitch) * torch.cos(roll), -torch.sin(roll)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([cos * ux - sin * uy, sin * ux + cos * uy, torch.cos(pitch) * torch.cos(roll)], dim=-1)


def tilt_error(quat: torch.Tensor, wanted_up: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between the body up axis and ``wanted_up`` (world, unit vectors)."""
    return torch.acos((thrust_axis(quat) * wanted_up).sum(-1).clamp(-1.0, 1.0))
