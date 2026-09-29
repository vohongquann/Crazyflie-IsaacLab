"""Terminations: flight envelope (RL cascade tasks) and landed / crashed (landing task, pad at the env origin)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from Drone_RL.uav.mdp.layers import FlightState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def left_flight_envelope(env: ManagerBasedRLEnv, min_height: float = 0.2, max_height: float = 4.0,
                         max_distance: float = 3.0, max_tilt: float = 1.57, asset_name: str = "robot") -> torch.Tensor:
    """Too low (near the ground), too high, too far from the environment origin, or tilted beyond ``max_tilt`` [rad]."""
    state = FlightState.of(env.scene[asset_name], env.scene.env_origins)
    height = state.position[:, 2]
    distance = state.position[:, :2].norm(dim=-1)
    tilt = torch.acos(state.rotation[:, 2, 2].clamp(-1.0, 1.0))
    return (height < min_height) | (height > max_height) | (distance > max_distance) | (tilt > max_tilt)


# ── Landing ─────────────────────────────────────────────────────────────────────────────

def pad_distance_and_height(env, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal distance [m] of the drone to the pad (at the env origin) and its height [m] above the pad."""
    position = env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins
    return position[:, :2].norm(dim=-1), position[:, 2]


def linear_speed(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_w.torch.norm(dim=-1)


def upright(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """cos of the tilt: 1 when level, 0 at 90 degrees (z component of the gravity direction in the body frame)."""
    return -env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, 2]


def landed(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), height: float = 0.06, radius: float = 0.15,
           max_speed: float = 0.6, min_upright: float = 0.9) -> torch.Tensor:
    """Success: at the pad height, on the marker, slow and level."""
    distance, z = pad_distance_and_height(env, asset_cfg)
    return (z < height) & (distance < radius) & (linear_speed(env, asset_cfg) < max_speed) & (upright(env, asset_cfg) > min_upright)


def crashed(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), height: float = 0.06, radius: float = 0.15,
            max_speed: float = 0.6, min_upright: float = 0.9, max_distance: float = 3.0,
            max_height: float = 3.0) -> torch.Tensor:
    """Failure: touched the ground without landing, flew away, or turned over."""
    distance, z = pad_distance_and_height(env, asset_cfg)
    touched = z < height
    lost = (distance > max_distance) | (z > max_height) | (upright(env, asset_cfg) < 0.3)
    return (touched & ~landed(env, asset_cfg, height, radius, max_speed, min_upright)) | lost
