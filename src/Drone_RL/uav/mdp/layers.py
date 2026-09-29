"""The four RL layers, the same cascade as the PID (pure torch, no Isaac).

    position --> velocity --> attitude --> rate --> 4 motor commands
    [p*, yaw*]   [v*, yaw*]   [a*, yaw*]   [w*, T*]

Every layer is a small network: it reads the command of the layer above plus the flight state, and writes the command
of the layer below (the rate layer writes the motor commands). The inputs and outputs are those of the PID layer of the
same name (``pid_control/``), so an RL layer can replace a PID layer one to one. The wanted yaw is not decided by the
position and velocity layers: it passes through them down to the attitude layer.

A layer is trained with the layers below it frozen (``rl_control/freeze.py``). The observation and the output scaling
written here are used both while a layer trains (``observations.layer_observation``) and when it runs frozen under a
higher layer (``actions/cascade_action.FrozenLayer``), so the frozen network sees exactly what it saw in training.

Policy output ``a`` is clipped to [-1, 1]; ``output`` scales it to physical units.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.propulsion import thrust_to_pwm
from isaaclab.utils.math import matrix_from_quat

GRAVITY = 9.81
WEIGHT_N = U.DRONE_MASS_TOTAL_KG * GRAVITY


@dataclass
class FlightState:
    position: torch.Tensor      # (N, 3) relative to the environment origin, world frame [m]
    rotation: torch.Tensor      # (N, 3, 3) body -> world
    velocity: torch.Tensor      # (N, 3) world frame [m/s]
    body_rates: torch.Tensor    # (N, 3) body frame [rad/s]

    @staticmethod
    def of(robot, env_origins: torch.Tensor) -> "FlightState":
        data = robot.data
        return FlightState(
            position=data.root_pos_w.torch - env_origins,
            rotation=matrix_from_quat(data.root_quat_w.torch),
            velocity=data.root_lin_vel_w.torch,
            body_rates=data.root_ang_vel_b.torch,
        )


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_of(rotation: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])


def tilt_error(rotation: torch.Tensor, wanted_acceleration: torch.Tensor) -> torch.Tensor:
    """Angle [rad] between the body up axis and the thrust direction that gives ``wanted_acceleration`` (world)."""
    force_per_mass = wanted_acceleration + torch.tensor([0.0, 0.0, GRAVITY], device=wanted_acceleration.device)
    wanted_up = force_per_mass / force_per_mass.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.acos((rotation[:, :, 2] * wanted_up).sum(-1).clamp(-1.0, 1.0))


class Layer:
    name: str
    hz: float                   # rate at which the network runs
    command_dim: int            # size of the command it follows
    action_dim: int             # size of the network output
    history: int                # past outputs in the observation
    below: str | None           # layer that follows its output (None: the motors)

    def observe(self, state: FlightState, command: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Observation (N, obs_dim). ``history`` (N, self.history, action_dim): past outputs, newest first."""
        raise NotImplementedError

    def output(self, action: torch.Tensor, command: torch.Tensor, state: FlightState) -> torch.Tensor:
        """Network output in [-1, 1] -> command of the layer below (or motor commands in [0, 1])."""
        raise NotImplementedError

    @property
    def obs_dim(self) -> int:
        state = FlightState(torch.zeros(1, 3), torch.eye(3).unsqueeze(0), torch.zeros(1, 3), torch.zeros(1, 3))
        command = torch.zeros(1, self.command_dim)
        return self.observe(state, command, torch.zeros(1, self.history, self.action_dim)).shape[-1]


class RateLayer(Layer):
    """[wanted body rates (3) rad/s, wanted total thrust (1) N] -> four motor commands in [0, 1]."""

    name, hz, command_dim, action_dim, history, below = "rate", 100.0, 4, 4, 4, None
    MOTOR_SCALE = 0.2            # motor command = feedforward(T*) + 0.2 a


    def observe(self, state, command, history):
        return torch.cat([
            command[:, :3] - state.body_rates,              # rate error (3)
            state.body_rates,                               # body rates (3)
            command[:, 3:4] / WEIGHT_N,                     # wanted thrust / weight (1)
            history.flatten(1),                             # last 4 outputs (16)
        ], dim=-1)

    def output(self, action, command, state):
        """Feedforward: the command that gives each motor T*/4 (inverse thrust curve), plus the learned correction, which
        makes the torques and fixes the thrust. The first run without the feedforward (hover + 0.5 a) did not learn."""
        a, b, c = U.CF_THRUST_COEF_G
        per_motor = (command[:, 3:4] / 4.0).expand(-1, 4)
        feedforward = thrust_to_pwm(per_motor, a, b, c, GRAVITY, U.CF_PWM_MAX) / U.CF_PWM_MAX
        return (feedforward + self.MOTOR_SCALE * action).clamp(0.0, 1.0)


class AttitudeLayer(Layer):
    """[wanted acceleration (3) m/s^2 world, wanted yaw (1) rad] -> [wanted body rates (3), wanted thrust (1)]."""

    name, hz, command_dim, action_dim, history, below = "attitude", 50.0, 4, 4, 2, "rate"
    RATE_SCALE = (6.0, 6.0, 3.0)    # [rad/s], the output limits of the PID attitude layer
    THRUST_SCALE = 0.5              # thrust correction, times the weight

    def observe(self, state, command, history):
        force_per_mass = command[:, :3] + torch.tensor([0.0, 0.0, GRAVITY], device=command.device)
        wanted_force_body = torch.einsum("nji,nj->ni", state.rotation, force_per_mass) / GRAVITY
        return torch.cat([
            wanted_force_body,                                          # (a* + g) in the body frame / g (3)
            wrap_angle(command[:, 3] - yaw_of(state.rotation)).unsqueeze(-1),   # yaw error (1)
            state.rotation[:, 2, :],                                    # world up axis seen from the body (3)
            state.body_rates,                                           # (3)
            history.flatten(1),                                         # last 2 outputs (8)
        ], dim=-1)

    def output(self, action, command, state):
        """Rates from the network. Thrust: feedforward m (a* + g) . z_body (the formula of the PID attitude layer) plus
        a learned correction of up to half the weight. The first run without the feedforward (thrust = mg (1 + a)) learned
        a wrong thrust and the drone fell under the velocity layer."""
        rates = action[:, :3] * torch.tensor(self.RATE_SCALE, device=action.device)
        force_per_mass = command[:, :3] + torch.tensor([0.0, 0.0, GRAVITY], device=command.device)
        feedforward = U.DRONE_MASS_TOTAL_KG * (force_per_mass * state.rotation[:, :, 2]).sum(-1, keepdim=True)
        thrust = (feedforward.clamp(min=0.0) + self.THRUST_SCALE * WEIGHT_N * action[:, 3:4]).clamp(min=0.0)
        return torch.cat([rates, thrust], dim=-1)


class VelocityLayer(Layer):
    """[wanted velocity (3) m/s world, wanted yaw (1)] -> [wanted acceleration (3) m/s^2 world, wanted yaw (1)]."""

    name, hz, command_dim, action_dim, history, below = "velocity", 50.0, 4, 3, 2, "attitude"
    ACCELERATION_SCALE = (7.0, 7.0, 6.0)    # [m/s^2], the output limits of the PID velocity layer

    def observe(self, state, command, history):
        return torch.cat([
            command[:, :3] - state.velocity,                # velocity error (3)
            state.velocity,                                 # (3)
            state.rotation[:, :, 2],                        # body up axis in the world: where the thrust points (3)
            history.flatten(1),                             # last 2 outputs (6)
        ], dim=-1)

    def output(self, action, command, state):
        acceleration = action * torch.tensor(self.ACCELERATION_SCALE, device=action.device)
        return torch.cat([acceleration, command[:, 3:4]], dim=-1)


class PositionLayer(Layer):
    """[target position (3) m, relative to the environment origin, wanted yaw (1)] -> [wanted velocity (3), yaw (1)]."""

    name, hz, command_dim, action_dim, history, below = "position", 50.0, 4, 3, 2, "velocity"
    VELOCITY_SCALE = 1.5          # [m/s] per axis, inside what the velocity layer was trained on

    def observe(self, state, command, history):
        return torch.cat([
            command[:, :3] - state.position,                # position error (3)
            state.velocity,                                 # (3)
            history.flatten(1),                             # last 2 outputs (6)
        ], dim=-1)

    def output(self, action, command, state):
        return torch.cat([self.VELOCITY_SCALE * action, command[:, 3:4]], dim=-1)


LAYERS: dict[str, Layer] = {layer.name: layer for layer in (RateLayer(), AttitudeLayer(), VelocityLayer(), PositionLayer())}
"""Training order: rate first, position last."""


def layers_below(name: str) -> list[Layer]:
    """The layers under ``name``, from the one right below down to the rate layer."""
    chain, below = [], LAYERS[name].below
    while below is not None:
        chain.append(LAYERS[below])
        below = LAYERS[below].below
    return chain
