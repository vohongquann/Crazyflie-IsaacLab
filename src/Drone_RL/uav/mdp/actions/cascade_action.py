"""Action of a layer in training: its output goes through the frozen layers below it, down to the motors.

    policy action (trained layer, env step)
        -> FrozenLayer(layer below)  every 1/hz of that layer
        -> ...
        -> FrozenLayer(rate)         every 1/500 s
        -> motor commands -> Propulsion (every physics step, 1 kHz)

For the rate layer the list of frozen layers is empty: its output are the motor commands.
With ``pid_rate`` the bottom of the chain is the PID rate controller (``pid_control/rate.py``) instead of the frozen rate
network: body rates and thrust -> torque -> mixer inverse -> motor commands.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.frozen_policy import load_frozen
from Drone_RL.uav.mdp.actions.mixer import force_allocation_inverse
from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
from Drone_RL.uav.mdp.actions.propulsion import Propulsion, thrust_to_pwm
from Drone_RL.uav.mdp.flight import GRAVITY
from Drone_RL.uav.mdp.layers import LAYERS, Layer, layers_below
from Drone_RL.uav.pid_control.rate import RateController

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
        self.command = torch.zeros(num_envs, layer.command_dim, device=device)      # what the layer above asked for
        self.output = torch.zeros(num_envs, output_dim(layer), device=device)

    def step(self, env, command: torch.Tensor) -> None:
        """Run the network once. ``CascadeAction.observing`` is this layer meanwhile, so the observation terms read its
        command and history."""
        self.command = command
        action = self.policy(self.layer.observe(env)).clamp(-1.0, 1.0)
        self.history.push(action)
        self.output = self.layer.output(action, command, env)

    def reset(self, env_ids) -> None:
        self.history.reset(env_ids)


class PIDRateLayer:
    """The PID rate controller in place of the frozen rate network: [wanted body rates (3), thrust (1)] -> torque
    (``RateController``) -> force of each motor (mixer inverse) -> motor commands in [0, 1]. Same interface as
    ``FrozenLayer``, so ``CascadeAction`` runs it at ``RateLayer.hz`` like a frozen layer."""

    def __init__(self, num_envs: int, physics_hz: float, device):
        self.layer = LAYERS["rate"]
        self.period = int(round(physics_hz / self.layer.hz))
        self.dt = self.period / physics_hz
        self.controller = RateController(device)
        self.allocation_inverse = force_allocation_inverse(device)
        self.output = torch.full((num_envs, 4), U.DRONE_HOVER_THROTTLE, device=device)

    def step(self, env, command: torch.Tensor) -> None:
        torque = self.controller.update(command[:, :3], isaac_mdp.base_ang_vel(env), self.dt)
        motor_forces = torch.cat([command[:, 3:4], torque], dim=-1) @ self.allocation_inverse.T
        a, b, c = U.CF_THRUST_COEF_G
        self.output = thrust_to_pwm(motor_forces, a, b, c, GRAVITY, U.CF_PWM_MAX) / U.CF_PWM_MAX

    def reset(self, env_ids) -> None:
        self.controller.reset(env_ids)


class CascadeAction(ActionTerm):
    cfg: "CascadeActionCfg"

    def __init__(self, cfg: "CascadeActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies(cfg.body_name)[0]
        physics_hz = 1.0 / env.physics_dt

        self.layer = LAYERS[cfg.layer]
        self.frozen = [PIDRateLayer(self.num_envs, physics_hz, self.device) if cfg.pid_rate and layer.name == "rate"
                       else FrozenLayer(layer, self.num_envs, physics_hz, cfg.frozen_dir, self.device)
                       for layer in layers_below(cfg.layer)]
        for layer in [self.layer] + [frozen.layer for frozen in self.frozen]:
            if abs(physics_hz / layer.hz - round(physics_hz / layer.hz)) > 1e-6:
                raise ValueError(f"the physics rate {physics_hz} Hz is not a multiple of the '{layer.name}' rate {layer.hz} Hz")

        self.history = History(self.num_envs, self.layer, self.device)
        self.observing = self           # the layer the observation terms read: this one, or a frozen one while it runs
        self._raw_actions = torch.zeros(self.num_envs, self.layer.action_dim, device=self.device)
        self._output = torch.zeros(self.num_envs, output_dim(self.layer), device=self.device)
        self._motor = torch.full((self.num_envs, 4), U.DRONE_HOVER_THROTTLE, device=self.device)
        self._propulsion = Propulsion(cfg, self.num_envs, self.device)
        # Physics steps since the start, never reset: a layer runs when it is a multiple of its period, like
        # RATE_DO_EXECUTE on the firmware tick. The periods need not divide the env step (velocity 5 over attitude 2).
        self._tick = 0
        self.velocity_at_step_start = torch.zeros(self.num_envs, 3, device=self.device)
        self._no_command = torch.zeros(self.num_envs, self.layer.command_dim, device=self.device)

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
        """Total thrust of the four motors now [N]."""
        return self._propulsion.force.sum(dim=1)

    @property
    def motor_commands(self) -> torch.Tensor:
        return self._motor

    @property
    def command(self) -> torch.Tensor:
        """What the layers above ask of the layer in training; zeros without a command term (landing task: the
        policy takes the place of the position layer, the wanted yaw is 0)."""
        if self.cfg.command_name is None:
            return self._no_command
        return self._env.command_manager.get_command(self.cfg.command_name)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self.history.reset(env_ids)
        for frozen in self.frozen:
            frozen.reset(env_ids)
        self._propulsion.reset(env_ids)
        self._motor[env_ids] = U.DRONE_HOVER_THROTTLE

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self.history.push(self._raw_actions)
        self._output[:] = self.layer.output(self._raw_actions, self.command, self._env)
        self.velocity_at_step_start[:] = self._robot.data.root_lin_vel_w.torch

    def apply_actions(self):
        command = self._output
        for frozen in self.frozen:
            if self._tick % frozen.period == 0:
                self.observing = frozen
                frozen.step(self._env, command)
                self.observing = self
            command = frozen.output
        self._motor[:] = command
        self._propulsion.step(self._motor * self.cfg.pwm_max, self._robot, self._body_id)
        self._tick += 1


@configclass
class CascadeActionCfg(MotorActionCfg):
    """Propulsion settings are those of ``MotorActionCfg`` (drag, motor asymmetry drawn every episode)."""

    class_type: type[ActionTerm] = CascadeAction
    layer: str = MISSING                  # "rate", "attitude", "velocity" or "position"
    command_name: str | None = "layer"    # None: no command term (zeros)
    frozen_dir: str = MISSING             # folder of the frozen <layer>.pt files (rl_control/frozen/rl/)
    pid_rate: bool = False                # rate below = PID rate controller instead of the frozen rate network
    spin_propellers: bool = False         # visual only; a joint write per environment every 2 ms slows training
