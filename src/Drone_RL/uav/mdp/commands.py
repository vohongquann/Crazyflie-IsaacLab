"""Command of the layer in training: what the layers above it would ask for.

The layers above a layer in training are the tuned PID layers (``pid_control/``), driven by a random target:

    random target position and yaw --PID position--> v* --PID velocity--> a* --> roll*, pitch*, yaw*, T* --PID attitude--> w*, T*
                                   ^ position layer   ^ velocity layer          ^ attitude layer                      ^ rate layer

The PID velocity layer writes the roll, pitch and thrust of the attitude layer.

The PID chain stops at the trained layer, whose command is what the PID above it outputs plus a random offset held for
0.3 to 1.5 s (``offset``), so the network also sees commands the PID would not give. The PID runs in closed loop on the
real flight, so the commands look like those of a real flight: large when the target jumps, small while it holds.

The target is fixed, or (position layer, ``path_types`` not empty) it runs on a closed horizontal path, a circle or a
figure 8 ("PathCommand" of IsaacLabUTE, ``balance_car/navigation/mdp/commands.py``): random radius, speed and direction,
and the path starts under the drone at every new target, so the layer follows the path and does not spend the episode
flying to it. The position command then also carries the velocity of the target (feed-forward).
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi

from Drone_RL.assets import DRONE_RL_ASSETS_DIR
from Drone_RL.uav.mdp.flight import tilt_error, up_axis_from_angles
from Drone_RL.uav.mdp.layers import LAYERS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PATH_TYPES = ("circle", "figure8")


def path_point(path_type: torch.Tensor, radius: torch.Tensor, spin: torch.Tensor, phase: torch.Tensor):
    """Point of the path relative to its center and its derivative by the phase, both (N, 2).

    ``path_type`` indexes ``PATH_TYPES``; ``spin`` is +1 or -1 (direction). The figure 8 is the lemniscate of Gerono, a
    8 lying down that passes the center twice per round.
    """
    is_circle = (path_type == PATH_TYPES.index("circle")).unsqueeze(-1)
    r, sin1, cos1, sin2, cos2 = radius, torch.sin(phase), torch.cos(phase), torch.sin(2.0 * phase), torch.cos(2.0 * phase)
    circle = torch.stack([r * cos1, r * spin * sin1], dim=-1)
    circle_slope = torch.stack([-r * sin1, r * spin * cos1], dim=-1)
    figure8 = torch.stack([r * sin1, r * spin * sin2 * 0.5], dim=-1)
    figure8_slope = torch.stack([r * cos1, r * spin * cos2], dim=-1)
    return torch.where(is_circle, circle, figure8), torch.where(is_circle, circle_slope, figure8_slope)


class LayerCommand(CommandTerm):
    cfg: "LayerCommandCfg"

    def __init__(self, cfg: "LayerCommandCfg", env: "ManagerBasedRLEnv"):
        self._markers = {}                      # before super().__init__, which already calls _set_debug_vis_impl
        super().__init__(cfg, env)
        self.layer = LAYERS[cfg.layer]
        # Imported here: pid_control imports mdp.actions (mixer, propulsion), so a module-level import would be circular.
        from Drone_RL.uav.pid_control.cascade import CascadePID

        self.pid = CascadePID(self.device)
        self.target = torch.zeros(self.num_envs, 4, device=self.device)          # position (3) and yaw (1)
        self.target_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal = torch.zeros(self.num_envs, 3, device=self.device)           # fixed target, when not on a path
        self._on_path = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._center = torch.zeros(self.num_envs, 2, device=self.device)         # the path: center, height, radius,
        self._height = torch.zeros(self.num_envs, device=self.device)            # speed, direction, phase, shape
        self._radius = torch.ones(self.num_envs, device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)
        self._spin = torch.ones(self.num_envs, device=self.device)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._path_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._path_codes = torch.tensor([PATH_TYPES.index(name) for name in cfg.path_types], dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)
        self.offset = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)
        self.offset_time_left = torch.zeros(self.num_envs, device=self.device)
        self._offset_scale = torch.zeros(self.layer.command_dim, device=self.device)     # cfg.offset, padded
        self._offset_scale[: len(cfg.offset)] = torch.tensor(cfg.offset, device=self.device)
        self.metrics["error"] = torch.zeros(self.num_envs, device=self.device)     # mean over the episode so far
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_steps = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self.pid.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.offset[ids] = 0.0
        self.offset_time_left[ids] = 0.0
        extras = super().reset(env_ids)
        self._error_sum[ids] = 0.0
        self._error_steps[ids] = 0.0
        # env.reset() observes without computing the commands, so the first observation would show a zero command. A
        # reset of every environment happens only there (or rarely in step, where the extra PID update is harmless).
        if env_ids is None or len(env_ids) == self.num_envs:
            self._update_command()
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        low = torch.tensor(self.cfg.target_low, device=self.device)
        high = torch.tensor(self.cfg.target_high, device=self.device)
        self._goal[env_ids] = low + torch.rand(n, 3, device=self.device) * (high - low)
        self.target[env_ids, 3] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.yaw_range
        if len(self._path_codes) == 0:
            return
        self._on_path[env_ids] = torch.rand(n, device=self.device) >= self.cfg.p_static
        self._radius[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.radius_range)
        self._speed[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.speed_range)
        self._spin[env_ids] = torch.where(torch.rand(n, device=self.device) < 0.5, 1.0, -1.0)
        self._path_type[env_ids] = self._path_codes[torch.randint(len(self._path_codes), (n,), device=self.device)]
        self._phase[env_ids] = 0.0
        # the center puts phase 0 of the path where the drone is now, at the height of the drone inside the target box
        offset, _ = path_point(self._path_type[env_ids], self._radius[env_ids], self._spin[env_ids], self._phase[env_ids])
        position = isaac_mdp.root_pos_w(self._env)[env_ids]
        self._center[env_ids] = position[:, :2] - offset
        self._height[env_ids] = position[:, 2].clamp(self.cfg.target_low[2], self.cfg.target_high[2])
        self._follow_path(0.0)

    def _follow_path(self, dt: float):
        """Move the target on its path by ``dt``; ``target`` and ``target_velocity`` follow the path or the fixed goal."""
        rate = self._speed / self._radius * self._spin                    # phase per second
        self._phase += rate * dt * self._on_path.float()
        offset, slope = path_point(self._path_type, self._radius, self._spin, self._phase)
        on = self._on_path.unsqueeze(-1)
        self.target[:, :2] = torch.where(on, self._center + offset, self._goal[:, :2])
        self.target[:, 2] = torch.where(self._on_path, self._height, self._goal[:, 2])
        self.target_velocity[:, :2] = torch.where(on, slope * rate.unsqueeze(-1), 0.0)
        self.target_velocity[:, 2] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        self._follow_path(dt)
        position = isaac_mdp.root_pos_w(self._env)
        velocity = isaac_mdp.root_lin_vel_w(self._env)
        quat = isaac_mdp.root_quat_w(self._env)
        self._draw_offsets(dt)
        yaw = self.target[:, 3:4]
        name = self.layer.name

        if name == "rate" and self.cfg.level_only:
            # The PID layers above hold the height and stop the drone (wanted velocity: 0 across, back to the target
            # height); no horizontal target, so no large tilt is asked for.
            vertical = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            wanted_velocity = self.pid.position.update(self.target[:, :3], position, dt) * vertical
            attitude_command = self.pid.velocity.update(torch.cat([wanted_velocity, yaw], dim=-1), velocity, dt)
            thrust, wanted_rates = self.pid.attitude.update(attitude_command, quat, dt)
            command = torch.cat([wanted_rates, thrust.unsqueeze(-1)], dim=-1)
        elif name == "position":
            command = torch.cat([self.target, self.target_velocity], dim=-1)
        else:
            wanted_velocity = self.pid.position.update(self.target[:, :3], position, dt)
            if name == "velocity":
                command = torch.cat([wanted_velocity, yaw], dim=-1)
            else:
                attitude_command = self.pid.velocity.update(torch.cat([wanted_velocity, yaw], dim=-1), velocity, dt)
                if name == "attitude":
                    command = attitude_command
                else:
                    thrust, wanted_rates = self.pid.attitude.update(attitude_command, quat, dt)
                    command = torch.cat([wanted_rates, thrust.unsqueeze(-1)], dim=-1)
        command = command + self.offset
        if name != "rate":
            yaw_index = 2 if name == "attitude" else 3
            command[:, yaw_index] = wrap_to_pi(command[:, yaw_index])
        self._command[:] = command

    def _draw_offsets(self, dt: float) -> None:
        """New random offset for the environments whose offset has expired; zero offset with probability ``p_zero``."""
        self.offset_time_left -= dt
        ids = (self.offset_time_left <= 0.0).nonzero().flatten()
        if len(ids) == 0:
            return
        offset = (torch.rand(len(ids), len(self._offset_scale), device=self.device) * 2.0 - 1.0) * self._offset_scale
        keep = (torch.rand(len(ids), 1, device=self.device) >= self.cfg.p_zero).float()
        self.offset[ids] = offset * keep
        low, high = self.cfg.offset_time
        self.offset_time_left[ids] = low + torch.rand(len(ids), device=self.device) * (high - low)

    def _update_metrics(self):
        c = self._command
        name = self.layer.name
        if name == "rate":
            error = (c[:, :3] - isaac_mdp.base_ang_vel(self._env)).norm(dim=-1)
        elif name == "attitude":
            error = tilt_error(isaac_mdp.root_quat_w(self._env), up_axis_from_angles(c[:, 0], c[:, 1], c[:, 2]))
        elif name == "velocity":
            error = (c[:, :3] - isaac_mdp.root_lin_vel_w(self._env)).norm(dim=-1)
        else:
            error = (c[:, :3] - isaac_mdp.root_pos_w(self._env)).norm(dim=-1)
        self._error_sum += error
        self._error_steps += 1.0
        self.metrics["error"] = self._error_sum / self._error_steps

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Target (position and velocity layers); velocity layer: arrows for the wanted and the actual velocity;
        position layer: the whole path the target runs on (blue dots)."""
        if debug_vis and not self._markers and self.cfg.layer in ("position", "velocity"):
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            from isaaclab.markers.config import SPHERE_MARKER_CFG

            def spheres(path, radius, color=None):
                cfg = SPHERE_MARKER_CFG.copy()
                cfg.prim_path = path
                cfg.markers["sphere"].radius = radius
                if color is not None:
                    cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                return VisualizationMarkers(cfg)

            def arrow(path, color, thickness):      # the arrow of Isaac Lab (``BLUE_ARROW_X_MARKER_CFG``), vendored
                return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={"arrow": sim_utils.UsdFileCfg(
                    usd_path=str(DRONE_RL_ASSETS_DIR / "markers" / "arrow_x.usd"),
                    scale=(1.0, thickness, thickness),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                )}))

            self._markers["target"] = spheres("/Visuals/Command/target", 0.03, color=(1.0, 0.15, 0.15))
            if self.cfg.layer == "velocity":
                self._markers["wanted_velocity"] = arrow("/Visuals/Command/wanted_velocity", (0.0, 1.0, 0.0), 0.06)
                self._markers["velocity"] = arrow("/Visuals/Command/velocity", (0.0, 0.0, 1.0), 0.1)
            else:
                self._markers["path"] = spheres("/Visuals/Command/path", 0.012, color=(0.15, 0.6, 1.0))
        for marker in self._markers.values():
            marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self._markers:
            return
        origins = self._env.scene.env_origins
        self._markers["target"].visualize(translations=self.target[:, :3] + origins)
        m = min(self.num_envs, self.VISUAL_ENVS)
        if "path" in self._markers:
            # a round of the path in ``PATH_POINTS`` dots; where the target is fixed the dots sit on it
            phase = torch.linspace(0.0, 2.0 * math.pi, self.PATH_POINTS, device=self.device).expand(m, -1)
            shape = lambda k: path_point(self._path_type[:m], self._radius[:m], self._spin[:m], phase[:, k])[0]
            xy = torch.stack([self._center[:m] + shape(k) for k in range(self.PATH_POINTS)], dim=1)       # (m, K, 2)
            xy = torch.where(self._on_path[:m, None, None], xy, self.target[:m, None, :2])
            z = self.target[:m, None, 2:3].expand(-1, self.PATH_POINTS, -1)
            points = torch.cat([xy, z], dim=-1) + origins[:m, None]
            self._markers["path"].visualize(translations=points.reshape(-1, 3))
        if "velocity" in self._markers:
            origin = isaac_mdp.root_pos_w(self._env)[:m] + origins[:m]
            self._arrow(self._markers["wanted_velocity"], origin + 0.05, self._command[:m, :3])
            self._arrow(self._markers["velocity"], origin, isaac_mdp.root_lin_vel_w(self._env)[:m])

    def _arrow(self, marker, origin: torch.Tensor, velocity: torch.Tensor) -> None:
        """Arrow from ``origin`` along ``velocity``, ``ARROW_PER_MS`` m long per m/s."""
        speed = velocity.norm(dim=-1)
        yaw = torch.atan2(velocity[:, 1], velocity[:, 0])
        pitch = -torch.asin((velocity[:, 2] / speed.clamp(min=1e-6)).clamp(-1.0, 1.0))
        scale = torch.tensor(marker.cfg.markers["arrow"].scale, device=self.device).repeat(len(speed), 1)
        scale[:, 0] *= speed * self.ARROW_PER_MS
        marker.visualize(origin, quat_from_euler_xyz(torch.zeros_like(yaw), pitch, yaw), scale)

    VISUAL_ENVS = 16        # arrows and paths are drawn for the first environments only
    PATH_POINTS = 32
    ARROW_PER_MS = 0.5      # [m] of arrow per m/s


@configclass
class LayerCommandCfg(CommandTermCfg):
    class_type: type = LayerCommand
    layer: str = MISSING
    asset_name: str = "robot"
    debug_vis: bool = True
    resampling_time_range: tuple[float, float] = (2.0, 4.0)     # new random target
    target_low: tuple[float, float, float] = (-1.0, -1.0, 1.0)  # [m] relative to the environment origin
    target_high: tuple[float, float, float] = (1.0, 1.0, 2.0)
    yaw_range: float = math.pi                                  # target yaw drawn in +-yaw_range [rad]
    level_only: bool = False
    """Rate layer: above it the PID layers only hold the height and stop the drone (no horizontal target). The random
    offsets then make most of the command. With the full PID chain the command sat at
    its 6 rad/s limit and the rate policy learned to hover and ignore it."""
    offset: tuple = (0.0, 0.0, 0.0, 0.0)                        # half-range of the random offset, units of the command
    offset_time: tuple[float, float] = (0.3, 1.5)               # [s] each offset is held this long
    p_zero: float = 0.3                                         # share of offsets that are zero
    path_types: tuple[str, ...] = ()
    """Shapes of the path of the target (``PATH_TYPES``), drawn at random; empty: a fixed target (position layer only)."""
    radius_range: tuple[float, float] = (0.3, 0.8)              # [m] of the path
    speed_range: tuple[float, float] = (0.3, 0.8)               # [m/s] the target runs on the path
    p_static: float = 0.2                                       # share of targets that stay fixed, when there are paths
