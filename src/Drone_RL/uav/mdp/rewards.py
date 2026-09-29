"""Rewards of the RL cascade tasks (one group per layer) and of the landing task.

Every tracking reward is ``exp(-error^2 / std^2)``: 1 on the command, 0 far from it, so staying in the air is worth more
than falling. The command comes from ``commands.LayerCommand``, the achieved values from ``actions.CascadeAction``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from Drone_RL.uav.mdp.layers import WEIGHT_N, FlightState, tilt_error, wrap_angle, yaw_of
from Drone_RL.uav.mdp.terminations import crashed, landed, linear_speed, pad_distance_and_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _action(env: ManagerBasedRLEnv, action_name: str = "cascade"):
    return env.action_manager.get_term(action_name)


def _state(env: ManagerBasedRLEnv, action_name: str = "cascade") -> FlightState:
    return _action(env, action_name).state()


def _command(env: ManagerBasedRLEnv, command_name: str = "layer") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _kernel(error_sq: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-error_sq / std**2)


def body_rate_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Rate layer: body rates against the wanted ones."""
    error = _command(env)[:, :3] - _state(env).body_rates
    return _kernel(error.square().sum(-1), std)


def body_rate_error_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rate layer: squared body rate error (use a negative weight). Unlike the kernels it still has a slope when the
    error is large, which is where a new rate policy starts."""
    return (_command(env)[:, :3] - _state(env).body_rates).square().sum(-1)


def thrust_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Rate layer: total motor thrust against the wanted one, both divided by the weight."""
    error = (_action(env).thrust - _command(env)[:, 3]) / WEIGHT_N
    return _kernel(error.square(), std)


def tilt_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Attitude layer: angle between the body up axis and the thrust direction the wanted acceleration needs."""
    return _kernel(tilt_error(_state(env).rotation, _command(env)[:, :3]).square(), std)


def tilt_error_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Attitude layer: squared tilt error [rad^2] (use a negative weight); keeps a slope where the kernel is flat."""
    return tilt_error(_state(env).rotation, _command(env)[:, :3]).square()


def yaw_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Attitude layer: heading against the wanted yaw."""
    error = wrap_angle(_command(env)[:, 3] - yaw_of(_state(env).rotation))
    return _kernel(error.square(), std)


def acceleration_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Attitude layer: acceleration over the last policy step (velocity change / step time) against the wanted one."""
    term = _action(env)
    acceleration = (term.state().velocity - term.velocity_at_step_start) / env.step_dt
    return _kernel((_command(env)[:, :3] - acceleration).square().sum(-1), std)


def velocity_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Velocity layer."""
    return _kernel((_command(env)[:, :3] - _state(env).velocity).square().sum(-1), std)


def velocity_error_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Velocity layer: squared velocity error (use a negative weight); keeps a slope where the kernels are flat."""
    return (_command(env)[:, :3] - _state(env).velocity).square().sum(-1)


def position_error_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position layer: squared position error (use a negative weight); keeps a slope where the kernels are flat."""
    return (_command(env)[:, :3] - _state(env).position).square().sum(-1)


def position_tracking(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Position layer."""
    return _kernel((_command(env)[:, :3] - _state(env).position).square().sum(-1), std)


def output_change(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Change of the network output since the last step, squared (use a negative weight)."""
    history = _action(env).history.values
    return (history[:, 0] - history[:, 1]).square().sum(-1)


def body_rates_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared body rates (use a negative weight)."""
    return _state(env).body_rates.square().sum(-1)


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


def landed_bonus(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return landed(env, asset_cfg).float()


def crashed_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return crashed(env, asset_cfg).float()
