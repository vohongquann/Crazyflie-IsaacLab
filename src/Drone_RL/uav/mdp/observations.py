"""Observations.

State list of Eschmann et al., arXiv:2404.07837, section IV-B (landing task): "position, orientation, linear velocity,
angular velocity, action history". Choices the paper does not fix (guide/06_aruco_landing.md, section 6.4):

    position        relative to the target point (3), world frame [m]
    orientation     rotation matrix body -> world, flattened (9)
    linear velocity world frame (3) [m/s]
    angular velocity body frame (3) [rad/s]
    action history  the last ``history`` motor commands of the policy, newest first (4 * history), each in [0, 1]

RL cascade (``layer_observation``): the observation of the layer in training, built by ``layers.py``.
Landing (``ArucoObservation``): what the downward camera says about the marker (``aruco.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from Drone_RL.uav.mdp.aruco import ArucoDetector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_relative_to_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                target_offset: tuple = (0.0, 0.0, 0.0)) -> torch.Tensor:
    """Position of the drone minus the target point (N, 3) [m]. The target is the env origin plus ``target_offset``."""
    position = env.scene[asset_cfg.name].data.root_pos_w.torch
    target = env.scene.env_origins + torch.tensor(target_offset, device=env.device)
    return position - target


def orientation_matrix(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Rotation matrix body -> world, flattened row by row (N, 9)."""
    quat = env.scene[asset_cfg.name].data.root_quat_w.torch
    return matrix_from_quat(quat).reshape(-1, 9)


def linear_velocity_world(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear velocity in the world frame (N, 3) [m/s]."""
    return env.scene[asset_cfg.name].data.root_lin_vel_w.torch


def angular_velocity_body(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocity in the body frame (N, 3) [rad/s]."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b.torch


class ActionHistory(ManagerTermBase):
    """The last ``history`` motor commands (N, 4 * history), newest first. Zero at the start of an episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._history = torch.zeros(env.num_envs, cfg.params["history"], 4, device=env.device)
        self._pushed_at_step = -1

    def reset(self, env_ids=None):
        self._history[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, history: int = 4, action_name: str = "motor") -> torch.Tensor:
        # Push once per environment step: the observation can be computed twice in one step (final observation, recorders).
        if env.common_step_counter != self._pushed_at_step:
            self._pushed_at_step = env.common_step_counter
            command = env.action_manager.get_term(action_name).raw_actions    # motor command in [0, 1], (N, 4)
            self._history = torch.roll(self._history, shifts=1, dims=1)
            self._history[:, 0] = command
        return self._history.reshape(env.num_envs, -1)


def layer_observation(env: ManagerBasedRLEnv, action_name: str = "cascade", command_name: str = "layer") -> torch.Tensor:
    """Observation of the RL cascade layer in training (``layers.py``): the same code runs the layer once it is frozen."""
    term = env.action_manager.get_term(action_name)
    return term.layer.observe(term.state(), env.command_manager.get_command(command_name), term.history.values)


class ArucoObservation(ManagerTermBase):
    """What the downward camera says about the marker: (u, v, size, found) (N, 4).

    u, v in [-1, 1] are the marker centre in the image (0 = straight below the drone), size is the marker side
    over the image width, found is 1 if the marker was detected in this frame. When it is not, u, v and size keep
    their last detected value (0 at the start of an episode), so that ``found = 0`` tells the policy they are old.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._detector = ArucoDetector()
        self._last = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids=None):
        self._last[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
        camera = env.scene.sensors[sensor_cfg.name]
        images = camera.data.output["rgb"]
        height, width = images.shape[1], images.shape[2]
        result = self._detector.detect(images.cpu().numpy())

        found = torch.as_tensor(result["found"], device=env.device)
        center = torch.as_tensor(result["center"], device=env.device)
        side = torch.as_tensor(result["side"], device=env.device)
        measured = torch.stack([
            (center[:, 0] - 0.5 * width) / (0.5 * width),        # u: + to the right of the image
            (center[:, 1] - 0.5 * height) / (0.5 * height),      # v: + towards the bottom of the image
            side / width,
        ], dim=-1)
        self._last = torch.where(found.unsqueeze(-1), measured, self._last)
        return torch.cat([self._last, found.float().unsqueeze(-1)], dim=-1)
