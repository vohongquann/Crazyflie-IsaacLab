"""The four layers of the cascade, the same as the PID (inputs and outputs of the PID layer of the same name).

    position --> velocity --> attitude --> rate --> 4 motor commands
    [p*, yaw*]   [v*, yaw*]   [roll*, pitch*,   [w*, T*]
                              yaw*, T*]

Each layer runs at its own rate and follows the command of the layer above. The wanted yaw is not decided by the
position and velocity layers: it passes through them down to the attitude layer. The velocity layer writes roll, pitch
and thrust (like the velocity controller of the Crazyflie firmware); the attitude layer writes body rates and passes the
thrust through to the rate layer.

``Layer.observation`` lists what the gain network of the layer observes (``observations.py``, functions of ``env`` like
those of Isaac Lab), followed by the history of its gains (``mdp/gains.py``). The env cfg of the layer lists the same
terms, and ``actions/gain_action.FrozenGainLayer`` calls them when the layer runs frozen under a higher layer, so the
frozen network sees exactly what it saw in training.
"""
from __future__ import annotations

import torch

from isaaclab.envs import mdp as isaac_mdp

from Drone_RL.uav.mdp import observations as obs
from Drone_RL.uav.mdp.actions.constants import ATTITUDE_HZ, POSITION_HZ, RATE_HZ, VELOCITY_HZ


class Layer:
    name: str
    hz: float                   # rate at which the layer runs
    command_dim: int            # size of the command it follows
    below: str | None           # layer that follows its output (None: the motors)
    observation: tuple          # observation terms, in the order of the observation vector (the history last)
    state_dim: int              # size of the terms before the history (a test checks it)

    def observe(self, env) -> torch.Tensor:
        """Observation of the layer the cascade is observing now (``GainCascadeAction.observing``)."""
        return torch.cat([term(env) for term in self.observation], dim=-1)


class RateLayer(Layer):
    """[wanted body rates (3) rad/s, wanted total thrust (1) N] -> four motor commands in [0, 1]."""

    name, hz, command_dim, below = "rate", RATE_HZ, 4, None
    observation = (obs.rate_error, isaac_mdp.base_ang_vel, obs.thrust_ratio, obs.action_history)
    state_dim = 3 + 3 + 1


class AttitudeLayer(Layer):
    """[wanted roll, pitch, yaw (3) rad, wanted total thrust (1) N] -> [wanted body rates (3), wanted thrust (1)]."""

    name, hz, command_dim, below = "attitude", ATTITUDE_HZ, 4, "rate"
    observation = (
        obs.attitude_error,
        obs.thrust_ratio,
        isaac_mdp.projected_gravity,
        isaac_mdp.base_ang_vel,
        obs.action_history,
    )
    state_dim = 3 + 1 + 3 + 3


class VelocityLayer(Layer):
    """[wanted velocity (3) m/s world, wanted yaw (1)] -> [wanted roll, pitch, yaw (3) rad, wanted total thrust (1) N]."""

    name, hz, command_dim, below = "velocity", VELOCITY_HZ, 4, "attitude"
    observation = (obs.velocity_error, isaac_mdp.root_lin_vel_w, obs.body_up_in_world, obs.action_history)
    state_dim = 3 + 3 + 3


class PositionLayer(Layer):
    """[target position (3) m, relative to the environment origin, wanted yaw (1), velocity of the target (3) m/s]
    -> [wanted velocity (3), yaw (1)]. The target velocity is 0 for a fixed target."""

    name, hz, command_dim, below = "position", POSITION_HZ, 7, "velocity"
    observation = (obs.position_error, isaac_mdp.root_lin_vel_w, obs.target_velocity, obs.action_history)
    state_dim = 3 + 3 + 3


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


def output_dim(layer: Layer) -> int:
    """Size of what ``layer`` writes: the command of the layer below, or the four motor commands."""
    return LAYERS[layer.below].command_dim if layer.below else 4
