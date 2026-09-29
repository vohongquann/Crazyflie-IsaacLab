"""Cascade of the four layers, each in its own file (pure torch, no Isaac).

    position.py --> velocity.py --> attitude.py --> rate.py --> to_pwm
    target pos      wanted vel      wanted accel     wanted      thrust, torque
                                                     rates       -> PWM of the 4 motors

Every layer takes what the layer above wants and returns what the layer below must follow, so a caller can enter the
chain anywhere, e.g. ``cascade.velocity.update(...)`` with a wanted velocity from a planner or a policy.
"""
import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.mixer import force_allocation_inverse
from Drone_RL.uav.mdp.actions.propulsion import thrust_to_pwm
from Drone_RL.uav.pid_control.attitude import GRAVITY, AttitudeController
from Drone_RL.uav.pid_control.position import PositionController
from Drone_RL.uav.pid_control.rate import RateController
from Drone_RL.uav.pid_control.velocity import VelocityController


class CascadePID:
    def __init__(self, device):
        self.position = PositionController(device)
        self.velocity = VelocityController(device)
        self.attitude = AttitudeController(device)
        self.rate = RateController(device)
        self.force_allocation_inverse = force_allocation_inverse(device)

    def reset(self, env_ids=None) -> None:
        for layer in (self.position, self.velocity, self.attitude, self.rate):
            layer.reset(env_ids)

    def to_pwm(self, thrust: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        """Thrust (N,) and torque (N, 3) -> force of each motor -> PWM (N, 4)."""
        wrench = torch.cat([thrust.unsqueeze(-1), torque], dim=-1)
        motor_forces = wrench @ self.force_allocation_inverse.T
        a, b, c = U.CF_THRUST_COEF_G
        return thrust_to_pwm(motor_forces, a, b, c, GRAVITY, U.CF_PWM_MAX)

    def step(self, target: torch.Tensor, pos: torch.Tensor, vel: torch.Tensor, quat: torch.Tensor,
             body_rates: torch.Tensor, dt: float) -> torch.Tensor:
        """All layers: target position -> PWM (N, 4)."""
        wanted_velocity = self.position.update(target, pos, dt)
        wanted_acceleration = self.velocity.update(wanted_velocity, vel, dt)
        thrust, wanted_rates = self.attitude.update(wanted_acceleration, quat, dt)
        torque = self.rate.update(wanted_rates, body_rates, dt)
        return self.to_pwm(thrust, torque)
