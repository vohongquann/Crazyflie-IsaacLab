"""Action of a gain layer in training: every layer is a network that writes the PID gains of its layer.

    policy action (9 gains, env step)                           trained layer
        -> PID of the layer, with these gains                   (``mdp/gains.py``)
        -> FrozenGainLayer(layer below)  every 1/hz of that layer: frozen network -> 9 gains -> PID with them
        -> ...
        -> FrozenGainLayer(rate)         every 1/500 s: its PID gives the torque -> mixer inverse -> motor commands
        -> Propulsion (every physics step, 1 kHz)

The same chain as ``cascade_action.CascadeAction``, with a PID computing the command of each layer from the gains its
network writes. The layers below are trained first and frozen (``frozen/gains/<layer>.pt``), bottom-up like the RL
cascade. With ``pid_below`` the layers below are the tuned PID instead (no frozen file needed).
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.cascade_action import History, PIDRateLayer, output_dim
from Drone_RL.uav.mdp.actions.frozen_policy import load_frozen_gains
from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
from Drone_RL.uav.mdp.actions.propulsion import Propulsion
from Drone_RL.uav.mdp.gains import GAIN_LAYERS
from Drone_RL.uav.mdp.layers import LAYERS, layers_below
from Drone_RL.uav.pid_control.attitude import AttitudeController
from Drone_RL.uav.pid_control.position import PositionController
from Drone_RL.uav.pid_control.velocity import VelocityController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PIDLayer:
    """A PID layer of ``pid_control/`` as a step of the chain: command of the layer -> command of the layer below.
    Same interface as ``PIDRateLayer`` (``period``, ``dt``, ``output``, ``step``, ``reset``, ``controller.pid``)."""

    controller_class: type

    def __init__(self, name: str, num_envs: int, physics_hz: float, device):
        self.layer = LAYERS[name]
        self.period = int(round(physics_hz / self.layer.hz))       # physics steps between two runs of the layer
        self.dt = self.period / physics_hz
        self.controller = self.controller_class(device)
        self.output = torch.zeros(num_envs, output_dim(self.layer), device=device)

    def reset(self, env_ids) -> None:
        self.controller.reset(env_ids)


class PIDAttitudeLayer(PIDLayer):
    """[roll, pitch, yaw, thrust] -> [wanted body rates (3), thrust (1)]; the thrust goes through."""

    controller_class = AttitudeController

    def step(self, env, command: torch.Tensor) -> None:
        thrust, wanted_rates = self.controller.update(command, isaac_mdp.root_quat_w(env), self.dt)
        self.output = torch.cat([wanted_rates, thrust.unsqueeze(-1)], dim=-1)


class PIDVelocityLayer(PIDLayer):
    """[wanted velocity (3), yaw] -> [roll, pitch, yaw, thrust]."""

    controller_class = VelocityController

    def step(self, env, command: torch.Tensor) -> None:
        self.output = self.controller.update(command, isaac_mdp.root_lin_vel_w(env), self.dt)


class PIDPositionLayer(PIDLayer):
    """[target position (3), yaw, target velocity (3)] -> [wanted velocity (3), yaw]; the target velocity is not used (a
    PID has no feed-forward here; the integral term is what follows a moving target)."""

    controller_class = PositionController

    def step(self, env, command: torch.Tensor) -> None:
        wanted_velocity = self.controller.update(command[:, :3], isaac_mdp.root_pos_w(env), self.dt)
        self.output = torch.cat([wanted_velocity, command[:, 3:4]], dim=-1)


def pid_layer(name: str, num_envs: int, physics_hz: float, device):
    if name == "rate":
        return PIDRateLayer(num_envs, physics_hz, device)
    classes = {"attitude": PIDAttitudeLayer, "velocity": PIDVelocityLayer, "position": PIDPositionLayer}
    return classes[name](name, num_envs, physics_hz, device)


class FrozenGainLayer:
    """A trained gain layer that no longer learns: observation -> frozen network -> 9 gains -> PID of the layer -> command
    of the layer below. ``CascadeAction.observing`` is this layer while it runs, so the observation terms read its command
    and its history, as in training."""

    def __init__(self, name: str, num_envs: int, physics_hz: float, frozen_dir, device):
        self.pid_layer = pid_layer(name, num_envs, physics_hz, device)
        self.gain_layer = GAIN_LAYERS[name]
        self.pid_layer.controller.pid.int_limit = self.gain_layer.int_limit.to(device)
        self.policy = load_frozen_gains(name, device, frozen_dir)
        self.layer = self.pid_layer.layer
        self.period = self.pid_layer.period
        self.history = History(num_envs, self.gain_layer, device)
        self.command = torch.zeros(num_envs, self.layer.command_dim, device=device)      # what the layer above asked for

    @property
    def output(self) -> torch.Tensor:
        return self.pid_layer.output

    def step(self, env, command: torch.Tensor) -> None:
        self.command = command
        action = self.policy(self.layer.observe(env)).clamp(-1.0, 1.0)
        self.history.push(action)
        pid = self.pid_layer.controller.pid
        pid.kp, pid.ki, pid.kd = self.gain_layer.gains(action)
        self.pid_layer.step(env, command)

    def reset(self, env_ids) -> None:
        self.history.reset(env_ids)
        self.pid_layer.reset(env_ids)


class GainCascadeAction(ActionTerm):
    cfg: "GainCascadeActionCfg"

    def __init__(self, cfg: "GainCascadeActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies(cfg.body_name)[0]
        physics_hz = 1.0 / env.physics_dt

        self.layer = LAYERS[cfg.layer]
        self.gain_layer = GAIN_LAYERS[cfg.layer]
        self.trained = pid_layer(cfg.layer, self.num_envs, physics_hz, self.device)
        self.trained.controller.pid.int_limit = self.gain_layer.int_limit.to(self.device)
        self.below = [pid_layer(layer.name, self.num_envs, physics_hz, self.device) if cfg.pid_below
                      else FrozenGainLayer(layer.name, self.num_envs, physics_hz, cfg.frozen_dir, self.device)
                      for layer in layers_below(cfg.layer)]
        for layer in [self.layer] + [below.layer for below in self.below]:
            if abs(physics_hz / layer.hz - round(physics_hz / layer.hz)) > 1e-6:
                raise ValueError(f"the physics rate {physics_hz} Hz is not a multiple of the '{layer.name}' rate {layer.hz} Hz")

        self.history = History(self.num_envs, self.gain_layer, self.device)
        self.observing = self           # the layer the observation terms read: this one, or a frozen one while it runs
        self._raw_actions = torch.zeros(self.num_envs, self.gain_layer.action_dim, device=self.device)
        self._gains = torch.zeros(self.num_envs, self.gain_layer.action_dim, device=self.device)
        self._output = torch.zeros(self.num_envs, output_dim(self.layer), device=self.device)
        self._motor = torch.full((self.num_envs, 4), U.DRONE_HOVER_THROTTLE, device=self.device)
        self._propulsion = Propulsion(cfg, self.num_envs, self.device)
        self._tick = 0      # physics steps since the start: a layer runs when it is a multiple of its period
        self.velocity_at_step_start = torch.zeros(self.num_envs, 3, device=self.device)
        self._no_command = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.gain_layer.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The gains now (N, 9): kp, ki, kd of the three axes."""
        return self._gains

    @property
    def thrust(self) -> torch.Tensor:
        """Total thrust of the four motors now [N]."""
        return self._propulsion.force.sum(dim=1)

    @property
    def motor_commands(self) -> torch.Tensor:
        return self._motor

    @property
    def command(self) -> torch.Tensor:
        """What the layers above ask of the layer in training."""
        if self.cfg.command_name is None:
            return self._no_command
        return self._env.command_manager.get_command(self.cfg.command_name)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self.history.reset(env_ids)
        self.trained.reset(env_ids)
        for below in self.below:
            below.reset(env_ids)
        self._propulsion.reset(env_ids)
        self._motor[env_ids] = U.DRONE_HOVER_THROTTLE

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self.history.push(self._raw_actions)
        pid = self.trained.controller.pid
        pid.kp, pid.ki, pid.kd = self.gain_layer.gains(self._raw_actions)
        self._gains[:] = torch.cat([pid.kp, pid.ki, pid.kd], dim=-1)
        self.trained.step(self._env, self.command)
        self._output[:] = self.trained.output
        self.velocity_at_step_start[:] = self._robot.data.root_lin_vel_w.torch

    def apply_actions(self):
        command = self._output
        for below in self.below:
            if self._tick % below.period == 0:
                self.observing = below
                below.step(self._env, command)
                self.observing = self
            command = below.output
        self._motor[:] = command
        self._propulsion.step(self._motor * self.cfg.pwm_max, self._robot, self._body_id)
        self._tick += 1


@configclass
class GainCascadeActionCfg(MotorActionCfg):
    """Propulsion settings are those of ``MotorActionCfg``."""

    class_type: type[ActionTerm] = GainCascadeAction
    layer: str = MISSING                  # "rate", "attitude", "velocity" or "position"
    command_name: str | None = "layer"    # None: no command term (zeros)
    frozen_dir: str = MISSING             # folder of the frozen <layer>.pt files (rl_control/frozen/gains/)
    pid_below: bool = False               # the layers below are the tuned PID instead of frozen gain networks
    spin_propellers: bool = False         # visual only; a joint write per environment every 2 ms slows training
