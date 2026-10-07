"""PID gains as the output of an RL layer (the ``-Gains-`` tasks).

The network of a layer does not write the command of the layer below; it writes the nine gains of the PID of that layer
(kp, ki, kd of the three axes), and the PID of ``pid_control/`` computes the command with them. Every other layer of the
cascade stays the tuned PID. After the idea of ``FPV-Drone-Tracking`` (``UAVAttitudeCascadePIDAction``: the network
schedules the gains of the outer loop, the inner loop is a fixed PID).

Action ``a`` in [-1, 1], one value per gain, order [kp x, y, z, ki x, y, z, kd x, y, z]. The tuned PID is the zero action:

    gain of the tuned PID not 0:   gain = nominal * 3^a              (a = -1 .. 1: a third to three times the tuned gain)
    gain of the tuned PID is 0:    gain = maximum * max(a, 0)        (a <= 0: the term is off)

``maximum`` is ``KI_PER_KP * kp`` [1/s] for ki and ``KD_PER_KP * kp`` [s] for kd.
"""
from __future__ import annotations

import torch

from Drone_RL.uav.mdp.layers import LAYERS
from Drone_RL.uav.pid_control.attitude import ATTITUDE_KP
from Drone_RL.uav.pid_control.position import POSITION_KP
from Drone_RL.uav.pid_control.rate import RATE_KD, RATE_KP
from Drone_RL.uav.pid_control.velocity import VELOCITY_INT_LIMIT, VELOCITY_KI, VELOCITY_KP

FACTOR = 3.0
KI_PER_KP = 0.3      # [1/s] largest ki of a layer whose tuned PID has none, relative to its kp
KD_PER_KP = 0.05     # [s] largest kd of such a layer, relative to its kp (a derivative time of 50 ms)

ZERO = (0.0, 0.0, 0.0)


class GainLayer:
    """The gains of one layer: the tuned values, and the map from the network output to gains. It also carries the two
    attributes ``History`` needs of a layer (``history``, ``action_dim``)."""

    history = 2          # past outputs in the observation
    action_dim = 9       # kp, ki, kd of three axes

    def __init__(self, name: str, kp: tuple, ki: tuple, kd: tuple, int_limit: tuple):
        self.name = name
        self.nominal = torch.tensor([kp, ki, kd])                                   # (kind, axis)
        kp_tensor = torch.tensor(kp)
        self.maximum = torch.stack([torch.zeros(3), KI_PER_KP * kp_tensor, KD_PER_KP * kp_tensor])
        self.int_limit = torch.tensor(int_limit)       # limit of the integral of the error (anti-windup)

    @property
    def obs_dim(self) -> int:
        """Observation of the layer (same terms as ``LAYERS[name]``) with the history of the gains in place of its own."""
        layer = LAYERS[self.name]
        return layer.obs_dim - layer.history * layer.action_dim + self.history * self.action_dim

    def gains(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Network output (N, 9) in [-1, 1] -> kp, ki, kd, each (N, 3)."""
        a = action.clamp(-1.0, 1.0).view(-1, 3, 3)
        nominal, maximum = self.nominal.to(a.device), self.maximum.to(a.device)
        gains = torch.where(nominal > 0.0, nominal * FACTOR ** a, maximum * a.clamp(min=0.0))
        return gains[:, 0], gains[:, 1], gains[:, 2]


GAIN_LAYERS: dict[str, GainLayer] = {
    "rate": GainLayer("rate", RATE_KP, ZERO, RATE_KD, int_limit=(0.5, 0.5, 0.5)),
    "attitude": GainLayer("attitude", ATTITUDE_KP, ZERO, ZERO, int_limit=(0.3, 0.3, 0.3)),
    "velocity": GainLayer("velocity", VELOCITY_KP, VELOCITY_KI, ZERO, int_limit=VELOCITY_INT_LIMIT),
    "position": GainLayer("position", POSITION_KP, ZERO, ZERO, int_limit=(0.5, 0.5, 0.5)),
}
