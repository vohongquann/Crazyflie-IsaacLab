"""Command of the layer in training: what the layers above it would ask for.

The layers above a layer in training are the tuned PID layers (``pid_control/``), driven by a random target:

    random target position and yaw --PID position--> v* --PID velocity--> a* --PID attitude--> (w*, T*)
                                   ^ position layer   ^ velocity layer    ^ attitude layer     ^ rate layer

The PID chain stops at the trained layer, whose command is what the PID above it outputs plus a random offset held for
0.3 to 1.5 s (``offset``), so the network also sees commands the PID would not give. The PID runs in closed loop on the
real flight, so the commands look like those of a real flight: large when the target jumps, small while it holds.
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from Drone_RL.uav.mdp.layers import LAYERS, FlightState, tilt_error, wrap_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LayerCommand(CommandTerm):
    cfg: "LayerCommandCfg"

    def __init__(self, cfg: "LayerCommandCfg", env: "ManagerBasedRLEnv"):
        self._marker = None                     # before super().__init__, which already calls _set_debug_vis_impl
        super().__init__(cfg, env)
        self.layer = LAYERS[cfg.layer]
        self.robot = env.scene[cfg.asset_name]
        # Imported here: pid_control imports mdp.actions (mixer, propulsion), so a module-level import would be circular.
        from Drone_RL.uav.pid_control.cascade import CascadePID

        self.pid = CascadePID(self.device)
        self.target = torch.zeros(self.num_envs, 4, device=self.device)          # position (3) and yaw (1)
        self._command = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)
        self.offset = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)
        self.offset_time_left = torch.zeros(self.num_envs, device=self.device)
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
        self.target[env_ids, :3] = low + torch.rand(n, 3, device=self.device) * (high - low)
        self.target[env_ids, 3] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.yaw_range

    def _update_command(self):
        dt = self._env.step_dt
        state = FlightState.of(self.robot, self._env.scene.env_origins)
        self._draw_offsets(dt)
        yaw = self.target[:, 3:4]
        name = self.layer.name

        if name == "rate" and self.cfg.level_only:
            # The PID layers above hold the height and stop the drone (wanted velocity: 0 across, back to the target
            # height); no horizontal target, so no large tilt is asked for.
            vertical = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            wanted_velocity = self.pid.position.update(self.target[:, :3], state.position, dt) * vertical
            wanted_acceleration = self.pid.velocity.update(wanted_velocity, state.velocity, dt)
            quat = self.robot.data.root_quat_w.torch
            thrust, wanted_rates = self.pid.attitude.update(wanted_acceleration, quat, dt, wanted_yaw=yaw[:, 0])
            command = torch.cat([wanted_rates, thrust.unsqueeze(-1)], dim=-1)
        elif name == "position":
            command = self.target.clone()
        else:
            wanted_velocity = self.pid.position.update(self.target[:, :3], state.position, dt)
            if name == "velocity":
                command = torch.cat([wanted_velocity, yaw], dim=-1)
            else:
                wanted_acceleration = self.pid.velocity.update(wanted_velocity, state.velocity, dt)
                if name == "attitude":
                    command = torch.cat([wanted_acceleration, yaw], dim=-1)
                else:
                    quat = self.robot.data.root_quat_w.torch
                    thrust, wanted_rates = self.pid.attitude.update(wanted_acceleration, quat, dt, wanted_yaw=yaw[:, 0])
                    command = torch.cat([wanted_rates, thrust.unsqueeze(-1)], dim=-1)
        command = command + self.offset
        if name != "rate":
            command[:, 3] = wrap_angle(command[:, 3])
        self._command[:] = command

    def _draw_offsets(self, dt: float) -> None:
        """New random offset for the environments whose offset has expired; zero offset with probability ``p_zero``."""
        self.offset_time_left -= dt
        ids = (self.offset_time_left <= 0.0).nonzero().flatten()
        if len(ids) == 0:
            return
        scale = torch.tensor(self.cfg.offset, device=self.device)
        offset = (torch.rand(len(ids), len(scale), device=self.device) * 2.0 - 1.0) * scale
        keep = (torch.rand(len(ids), 1, device=self.device) >= self.cfg.p_zero).float()
        self.offset[ids] = offset * keep
        low, high = self.cfg.offset_time
        self.offset_time_left[ids] = low + torch.rand(len(ids), device=self.device) * (high - low)

    def _update_metrics(self):
        state = FlightState.of(self.robot, self._env.scene.env_origins)
        c = self._command
        name = self.layer.name
        if name == "rate":
            error = (c[:, :3] - state.body_rates).norm(dim=-1)
        elif name == "attitude":
            error = tilt_error(state.rotation, c[:, :3])
        elif name == "velocity":
            error = (c[:, :3] - state.velocity).norm(dim=-1)
        else:
            error = (c[:, :3] - state.position).norm(dim=-1)
        self._error_sum += error
        self._error_steps += 1.0
        self.metrics["error"] = self._error_sum / self._error_steps

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and self._marker is None and self.cfg.layer in ("position", "velocity"):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import SPHERE_MARKER_CFG

            cfg = SPHERE_MARKER_CFG.copy()
            cfg.prim_path = "/Visuals/Command/target"
            cfg.markers["sphere"].radius = 0.03
            self._marker = VisualizationMarkers(cfg)
        if self._marker is not None:
            self._marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self._marker is not None:
            self._marker.visualize(translations=self.target[:, :3] + self._env.scene.env_origins)


@configclass
class LayerCommandCfg(CommandTermCfg):
    class_type: type = LayerCommand
    layer: str = MISSING
    asset_name: str = "robot"
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
