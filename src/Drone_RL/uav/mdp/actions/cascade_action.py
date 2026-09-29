"""Action of a layer in training: its output goes through the frozen layers below it, down to the motors.

    policy action (trained layer, env step)
        -> FrozenLayer(layer below)  every 1/hz of that layer
        -> ...
        -> FrozenLayer(rate)         every 1/100 s
        -> motor commands -> Propulsion (every physics step, 500 Hz)

For the rate layer the list of frozen layers is empty: its output are the motor commands.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.frozen_policy import load_frozen
from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
from Drone_RL.uav.mdp.actions.propulsion import Propulsion
from Drone_RL.uav.mdp.layers import LAYERS, FlightState, Layer, layers_below

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def output_dim(layer: Layer) -> int:
    """Size of what ``layer`` writes: the command of the layer below, or the four motor commands."""
    return LAYERS[layer.below].command_dim if layer.below else 4


class History:
    """Last outputs of a layer (N, length, action_dim), newest first; the part of the observation both uses share."""

    def __init__(self, num_envs: int, layer: Layer, device):
        self.values = torch.zeros(num_envs, layer.history, layer.action_dim, device=device)

    def push(self, action: torch.Tensor) -> None:
        self.values = torch.roll(self.values, shifts=1, dims=1)
        self.values[:, 0] = action

    def reset(self, env_ids) -> None:
        self.values[env_ids] = 0.0


class FrozenLayer:
    """A trained layer that is no longer learning: observation -> frozen network -> command of the layer below."""

    def __init__(self, layer: Layer, num_envs: int, physics_hz: float, frozen_dir, device):
        self.layer = layer
        self.policy = load_frozen(layer.name, device, frozen_dir)
        self.period = int(round(physics_hz / layer.hz))       # physics steps between two runs of the network
        self.history = History(num_envs, layer, device)
        self.output = torch.zeros(num_envs, output_dim(layer), device=device)

    def step(self, state: FlightState, command: torch.Tensor) -> None:
        action = self.policy(self.layer.observe(state, command, self.history.values)).clamp(-1.0, 1.0)
        self.history.push(action)
        self.output = self.layer.output(action, command, state)

    def reset(self, env_ids) -> None:
        self.history.reset(env_ids)


class CascadeAction(ActionTerm):
    cfg: "CascadeActionCfg"

    def __init__(self, cfg: "CascadeActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies(cfg.body_name)[0]
        self._physics_dt = env.physics_dt
        physics_hz = 1.0 / env.physics_dt

        self.layer = LAYERS[cfg.layer]
        self.frozen = [FrozenLayer(layer, self.num_envs, physics_hz, cfg.frozen_dir, self.device)
                       for layer in layers_below(cfg.layer)]
        for frozen in self.frozen:
            if (env.cfg.decimation % frozen.period) != 0:
                raise ValueError(f"decimation {env.cfg.decimation} is not a multiple of the '{frozen.layer.name}' period")

        self.history = History(self.num_envs, self.layer, self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.layer.action_dim, device=self.device)
        self._output = torch.zeros(self.num_envs, output_dim(self.layer), device=self.device)
        self._motor = torch.full((self.num_envs, 4), U.DRONE_HOVER_THROTTLE, device=self.device)
        self._propulsion = Propulsion(cfg, self.num_envs, self.device)
        self._tick = 0
        self.velocity_at_step_start = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.layer.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._output

    @property
    def thrust(self) -> torch.Tensor:
        """Total thrust of the four motors now (after the motor lag) [N]."""
        return self._propulsion.force.sum(dim=1)

    @property
    def motor_commands(self) -> torch.Tensor:
        return self._motor

    def state(self) -> FlightState:
        return FlightState.of(self._robot, self._env.scene.env_origins)

    def command(self) -> torch.Tensor:
        return self._env.command_manager.get_command(self.cfg.command_name)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self.history.reset(env_ids)
        for frozen in self.frozen:
            frozen.reset(env_ids)
        self._propulsion.reset(env_ids)
        # Start every episode with the motors at hover thrust, as if the drone had been flying.
        self._propulsion.force[env_ids] = U.DRONE_HOVER_THRUST_N / 4.0
        self._motor[env_ids] = U.DRONE_HOVER_THROTTLE

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self.history.push(self._raw_actions)
        self._output[:] = self.layer.output(self._raw_actions, self.command(), self.state())
        self.velocity_at_step_start[:] = self._robot.data.root_lin_vel_w.torch
        self._tick = 0

    def apply_actions(self):
        command = self._output
        if self.frozen:
            state = self.state()
            for frozen in self.frozen:
                if self._tick % frozen.period == 0:
                    frozen.step(state, command)
                command = frozen.output
        self._motor[:] = command
        self._propulsion.step(self._motor * self.cfg.pwm_max, self._robot, self._body_id, self._physics_dt)
        self._tick += 1


@configclass
class CascadeActionCfg(MotorActionCfg):
    """Propulsion settings are those of ``MotorActionCfg`` (motor lag, drag, motor asymmetry drawn every episode)."""

    class_type: type[ActionTerm] = CascadeAction
    layer: str = MISSING                  # "rate", "attitude", "velocity" or "position"
    command_name: str = "layer"
    frozen_dir: str = MISSING             # folder of the frozen <layer>.pt files (rl_control/frozen/)
    spin_propellers: bool = False         # visual only; a joint write per environment every 2 ms slows training
