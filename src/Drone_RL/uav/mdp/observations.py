"""Observation terms.

RL cascade: the observation of a layer, one function per term, written like the ones of Isaac Lab (``env`` in, tensor
out). The robot is read through Isaac Lab (``base_ang_vel``, ``root_lin_vel_w``, ``root_pos_w``, ``root_quat_w``,
``projected_gravity``); the command and the last outputs come from ``CascadeAction.observing``: the layer in training,
or the frozen layer that the cascade is running at that moment. So a frozen layer sees what it saw in training, with the
same functions. Each layer lists its terms in ``layers.py`` (``Layer.observation``) and in its env cfg.

Landing: the state list of Eschmann et al., arXiv:2404.07837, section IV-B ("position, orientation, linear velocity,
angular velocity, action history"), with the choices the paper does not fix (guide/06_aruco_landing.md, section 6.4):

    position        relative to the target point (3), world frame [m]       ``position_relative_to_target``
    orientation     quaternion (x, y, z, w), world frame, real part non-negative (4)    Isaac Lab ``root_quat_w``
    linear velocity world frame (3) [m/s]                                   Isaac Lab ``root_lin_vel_w``
    angular velocity body frame (3) [rad/s]                                 Isaac Lab ``base_ang_vel``
    action history  the last motor commands (4 * history_length), each in [0, 1]    Isaac Lab ``last_action``

plus what the downward camera says about the marker (``ArucoObservation``, ``aruco.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi

from Drone_RL.uav.mdp.aruco import ArucoDetector
from Drone_RL.uav.mdp.flight import GRAVITY, WEIGHT_N, thrust_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── RL cascade ──────────────────────────────────────────────────────────────────────────
# ``action_name``: the cascade action term; its ``observing`` layer has ``command`` (N, command_dim) and ``history``.

def rate_error(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Wanted body rates minus body rates (3)."""
    layer = env.action_manager.get_term(action_name).observing
    return layer.command[:, :3] - isaac_mdp.base_ang_vel(env)


def velocity_error(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Wanted velocity minus velocity (3)."""
    layer = env.action_manager.get_term(action_name).observing
    return layer.command[:, :3] - isaac_mdp.root_lin_vel_w(env)


def position_error(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Target position minus position (3)."""
    layer = env.action_manager.get_term(action_name).observing
    return layer.command[:, :3] - isaac_mdp.root_pos_w(env)


def thrust_ratio(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Wanted total thrust / weight (1)."""
    layer = env.action_manager.get_term(action_name).observing
    return layer.command[:, 3:4] / WEIGHT_N


def wanted_force_body(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Wanted force per mass (a* + g) in the body frame, divided by g (3)."""
    layer = env.action_manager.get_term(action_name).observing
    force_per_mass = layer.command[:, :3] + torch.tensor([0.0, 0.0, GRAVITY], device=env.device)
    return quat_apply_inverse(isaac_mdp.root_quat_w(env), force_per_mass) / GRAVITY


def yaw_error(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Wanted yaw minus yaw, wrapped to [-pi, pi] (1)."""
    layer = env.action_manager.get_term(action_name).observing
    yaw = euler_xyz_from_quat(isaac_mdp.root_quat_w(env))[2]
    return wrap_to_pi(layer.command[:, 3] - yaw).unsqueeze(-1)


def body_up_in_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body up axis in the world frame: where the thrust points (3)."""
    return thrust_axis(isaac_mdp.root_quat_w(env))


def action_history(env: ManagerBasedRLEnv, action_name: str = "cascade") -> torch.Tensor:
    """Last outputs of the layer, newest first (history * action_dim)."""
    return env.action_manager.get_term(action_name).observing.history.values.flatten(1)


# ── Landing ─────────────────────────────────────────────────────────────────────────────

def position_relative_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_offset: tuple = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Position of the drone minus the target point (N, 3) [m]. The target is the env origin plus ``target_offset``."""
    position = env.scene[asset_cfg.name].data.root_pos_w.torch
    target = env.scene.env_origins + torch.tensor(target_offset, device=env.device)
    return position - target


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
