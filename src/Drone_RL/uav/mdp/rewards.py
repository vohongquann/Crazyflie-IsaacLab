"""Rewards of the RL cascade tasks and of the landing task.

Cascade: the error to the command of the layer in training, written like Isaac Lab's ``position_command_error`` and
``track_lin_vel_xy_exp``. The ``_l2`` terms (negative weight) keep a slope when the error is large, which is where a new
policy starts; the ``_exp`` terms are ``exp(-error^2 / std^2)``: 1 on the command, 0 far from it. The command comes
from ``commands.LayerCommand``, the thrust and the velocity at the start of the step from ``actions.CascadeAction``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from Drone_RL.uav.mdp.flight import WEIGHT_N, tilt_error
from Drone_RL.uav.mdp.terminations import linear_speed, pad_distance_and_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── RL cascade ──────────────────────────────────────────────────────────────────────────
# ``command_name``: the command term of the layer in training; ``action_name``: the cascade action term.

def rate_error_l2(env: ManagerBasedRLEnv, command_name: str = "layer") -> torch.Tensor:
    """Squared error of the body rates to the wanted ones (rate layer)."""
    command = env.command_manager.get_command(command_name)
    return (command[:, :3] - isaac_mdp.base_ang_vel(env)).square().sum(-1)


def velocity_error_l2(env: ManagerBasedRLEnv, command_name: str = "layer") -> torch.Tensor:
    """Squared error of the velocity to the wanted one (velocity layer)."""
    command = env.command_manager.get_command(command_name)
    return (command[:, :3] - isaac_mdp.root_lin_vel_w(env)).square().sum(-1)


def position_error_l2(env: ManagerBasedRLEnv, command_name: str = "layer") -> torch.Tensor:
    """Squared error of the position to the target (position layer)."""
    command = env.command_manager.get_command(command_name)
    return (command[:, :3] - isaac_mdp.root_pos_w(env)).square().sum(-1)


def tilt_error_l2(env: ManagerBasedRLEnv, command_name: str = "layer") -> torch.Tensor:
    """Squared angle between the body up axis and the thrust direction the wanted acceleration needs (attitude layer)."""
    command = env.command_manager.get_command(command_name)
    return tilt_error(isaac_mdp.root_quat_w(env), command[:, :3]).square()


def yaw_error_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "layer") -> torch.Tensor:
    """Heading against the wanted yaw (attitude layer)."""
    command = env.command_manager.get_command(command_name)
    yaw = euler_xyz_from_quat(isaac_mdp.root_quat_w(env))[2]
    return torch.exp(-wrap_to_pi(command[:, 3] - yaw).square() / std**2)


def acceleration_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "layer",
    action_name: str = "cascade",
) -> torch.Tensor:
    """Acceleration over the last policy step (velocity change / step time) against the wanted one (attitude layer)."""
    command = env.command_manager.get_command(command_name)
    velocity_before = env.action_manager.get_term(action_name).velocity_at_step_start
    achieved = (isaac_mdp.root_lin_vel_w(env) - velocity_before) / env.step_dt
    return torch.exp(-(command[:, :3] - achieved).square().sum(-1) / std**2)


def thrust_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "layer",
    action_name: str = "cascade",
) -> torch.Tensor:
    """Total motor thrust against the wanted one, both divided by the weight (rate layer)."""
    command = env.command_manager.get_command(command_name)
    thrust = env.action_manager.get_term(action_name).thrust
    return torch.exp(-((thrust - command[:, 3]) / WEIGHT_N).square() / std**2)


def body_rates_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared body rates, the three axes (Isaac Lab's ``ang_vel_xy_l2`` leaves out yaw)."""
    return isaac_mdp.base_ang_vel(env).square().sum(-1)


# ── Landing ─────────────────────────────────────────────────────────────────────────────

def alignment(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 0.25) -> torch.Tensor:
    """exp(-d / scale): 1 straight above the pad, small far from it."""
    distance, _ = pad_distance_and_height(env, asset_cfg)
    return torch.exp(-distance / scale)


def descent_over_pad(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 0.25,
                     start_height: float = 1.4) -> torch.Tensor:
    """Alignment times the height already lost: descending only pays when the drone is above the pad."""
    distance, z = pad_distance_and_height(env, asset_cfg)
    return torch.exp(-distance / scale) * (1.0 - (z / start_height).clamp(0.0, 1.0))


def touchdown_speed_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 0.3) -> torch.Tensor:
    """Speed squared, weighted by closeness to the ground: it asks for a soft touchdown."""
    _, z = pad_distance_and_height(env, asset_cfg)
    return linear_speed(env, asset_cfg).square() * torch.exp(-z.clamp(min=0.0) / scale)
