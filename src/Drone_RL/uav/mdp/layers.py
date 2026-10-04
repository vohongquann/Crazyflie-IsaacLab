"""The four RL layers, the same cascade as the PID (inputs and outputs of the PID layer of the same name).

    position --> velocity --> attitude --> rate --> 4 motor commands
    [p*, yaw*]   [v*, yaw*]   [a*, yaw*]   [w*, T*]

Every layer is a small network: it reads the command of the layer above plus the flight state, and writes the command
of the layer below (the rate layer writes the motor commands). So an RL layer can replace a PID layer one to one. The
wanted yaw is not decided by the position and velocity layers: it passes through them down to the attitude layer.

A layer is trained with the layers below it frozen (``rl_control/frozen/``). ``Layer.observation`` lists its observation
terms (``observations.py``, functions of ``env`` like those of Isaac Lab); the env cfg of the layer lists the same ones,
and ``actions/cascade_action.FrozenLayer`` calls them when the layer runs frozen under a higher layer, so the frozen
network sees exactly what it saw in training.

Policy output ``a`` is clipped to [-1, 1]; ``output`` scales it to physical units.
"""
from __future__ import annotations

import torch

from isaaclab.envs import mdp as isaac_mdp

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp import observations as obs
from Drone_RL.uav.mdp.actions.constants import ATTITUDE_HZ, POSITION_HZ, RATE_HZ, VELOCITY_HZ
from Drone_RL.uav.mdp.actions.propulsion import thrust_to_pwm
from Drone_RL.uav.mdp.flight import GRAVITY, WEIGHT_N, thrust_axis


class Layer:
    name: str
    hz: float                   # rate at which the network runs
    command_dim: int            # size of the command it follows
    action_dim: int             # size of the network output
    history: int                # past outputs in the observation
    below: str | None           # layer that follows its output (None: the motors)
    observation: tuple          # observation terms, in the order of the observation vector
    obs_dim: int                # sum of their sizes (a test checks it)

    def observe(self, env) -> torch.Tensor:
        """Observation (N, obs_dim) of the layer the cascade is observing now (``CascadeAction.observing``)."""
        return torch.cat([term(env) for term in self.observation], dim=-1)

    def output(self, action: torch.Tensor, command: torch.Tensor, env) -> torch.Tensor:
        """Network output in [-1, 1] -> command of the layer below (or motor commands in [0, 1])."""
        raise NotImplementedError


class RateLayer(Layer):
    """[wanted body rates (3) rad/s, wanted total thrust (1) N] -> four motor commands in [0, 1]."""

    name, hz, command_dim, action_dim, history, below = "rate", RATE_HZ, 4, 4, 4, None
    observation = (obs.rate_error, isaac_mdp.base_ang_vel, obs.thrust_ratio, obs.action_history)
    obs_dim = 3 + 3 + 1 + 16
    MOTOR_SCALE = 0.2            # motor command = feedforward(T*) + 0.2 a

    def output(self, action, command, env):
        """Feedforward: the command that gives each motor T*/4 (inverse thrust curve), plus the learned correction, which
        makes the torques and fixes the thrust. The first run without the feedforward (hover + 0.5 a) did not learn."""
        a, b, c = U.CF_THRUST_COEF_G
        per_motor = (command[:, 3:4] / 4.0).expand(-1, 4)
        feedforward = thrust_to_pwm(per_motor, a, b, c, GRAVITY, U.CF_PWM_MAX) / U.CF_PWM_MAX
        return (feedforward + self.MOTOR_SCALE * action).clamp(0.0, 1.0)


class AttitudeLayer(Layer):
    """[wanted acceleration (3) m/s^2 world, wanted yaw (1) rad] -> [wanted body rates (3), wanted thrust (1)]."""

    name, hz, command_dim, action_dim, history, below = "attitude", ATTITUDE_HZ, 4, 4, 2, "rate"
    observation = (
        obs.wanted_force_body,
        obs.yaw_error,
        isaac_mdp.projected_gravity,
        isaac_mdp.base_ang_vel,
        obs.action_history,
    )
    obs_dim = 3 + 1 + 3 + 3 + 8
    RATE_SCALE = (6.0, 6.0, 3.0)    # [rad/s], the output limits of the PID attitude layer
    THRUST_SCALE = 0.5              # thrust correction, times the weight

    def output(self, action, command, env):
        """Rates from the network. Thrust: feedforward m (a* + g) . z_body (the formula of the PID attitude layer) plus
        a learned correction of up to half the weight. The first run without the feedforward (thrust = mg (1 + a)) learned
        a wrong thrust and the drone fell under the velocity layer."""
        rates = action[:, :3] * torch.tensor(self.RATE_SCALE, device=action.device)
        force_per_mass = command[:, :3] + torch.tensor([0.0, 0.0, GRAVITY], device=command.device)
        up = thrust_axis(isaac_mdp.root_quat_w(env))
        feedforward = U.DRONE_MASS_TOTAL_KG * (force_per_mass * up).sum(-1, keepdim=True)
        thrust = (feedforward.clamp(min=0.0) + self.THRUST_SCALE * WEIGHT_N * action[:, 3:4]).clamp(min=0.0)
        return torch.cat([rates, thrust], dim=-1)


class VelocityLayer(Layer):
    """[wanted velocity (3) m/s world, wanted yaw (1)] -> [wanted acceleration (3) m/s^2 world, wanted yaw (1)]."""

    name, hz, command_dim, action_dim, history, below = "velocity", VELOCITY_HZ, 4, 3, 2, "attitude"
    observation = (obs.velocity_error, isaac_mdp.root_lin_vel_w, obs.body_up_in_world, obs.action_history)
    obs_dim = 3 + 3 + 3 + 6
    ACCELERATION_SCALE = (7.0, 7.0, 6.0)    # [m/s^2], the output limits of the PID velocity layer

    def output(self, action, command, env):
        acceleration = action * torch.tensor(self.ACCELERATION_SCALE, device=action.device)
        return torch.cat([acceleration, command[:, 3:4]], dim=-1)


class PositionLayer(Layer):
    """[target position (3) m, relative to the environment origin, wanted yaw (1)] -> [wanted velocity (3), yaw (1)]."""

    name, hz, command_dim, action_dim, history, below = "position", POSITION_HZ, 4, 3, 2, "velocity"
    observation = (obs.position_error, isaac_mdp.root_lin_vel_w, obs.action_history)
    obs_dim = 3 + 3 + 6
    VELOCITY_SCALE = 1.5          # [m/s] per axis, inside what the velocity layer was trained on

    def output(self, action, command, env):
        return torch.cat([self.VELOCITY_SCALE * action, command[:, 3:4]], dim=-1)


LAYERS: dict[str, Layer] = {
    layer.name: layer for layer in (RateLayer(), AttitudeLayer(), VelocityLayer(), PositionLayer())
}
"""Training order: rate first, position last."""


def layers_below(name: str) -> list[Layer]:
    """The layers under ``name``, from the one right below down to the rate layer."""
    chain, below = [], LAYERS[name].below
    while below is not None:
        chain.append(LAYERS[below])
        below = LAYERS[below].below
    return chain
