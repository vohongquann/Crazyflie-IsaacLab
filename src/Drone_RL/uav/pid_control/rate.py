"""Rate layer: body rate error -> torque. Innermost loop: tune it first.

Test: the rate layer flies the attitude (the setpoint steps the roll, pitch and yaw rate, square waves that return the
drone to level); the thrust only holds the height at 3 m, so the drone does not fall during the test.

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/rate.py --live --viz kit
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/rate.py            # no window, PNG only
"""
import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.pid_control.attitude import GRAVITY
from Drone_RL.uav.pid_control.flight_test import LayerTest, run, step_table
from Drone_RL.uav.pid_control.pid import PID


class RateController:
    def __init__(self, device, inertia: tuple = U.DRONE_INERTIA_DIAG):
        self.inertia = torch.tensor(inertia, dtype=torch.float32, device=device)
        kp = torch.tensor([28.4, 28.4, 9.7], device=device)          # (roll, pitch, yaw), output [rad/s^2]
        kd = torch.tensor([0.30, 0.30, 0.14], device=device)
        self.pid = PID(kp=kp, kd=kd)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, wanted_rates: torch.Tensor, body_rates: torch.Tensor, dt: float) -> torch.Tensor:
        """Wanted and measured body rates ((N, 3) [rad/s]) -> torque [N m]: inertia * angular acceleration."""
        return self.inertia * self.pid.update(wanted_rates - body_rates, dt)


# (t_start [s], roll rate, pitch rate, yaw rate) in deg/s
STEPS = [
    (0.0, 0, 0, 0),
    (1.0, 30, 0, 0), (1.5, -30, 0, 0), (2.0, 0, 0, 0),
    (3.0, 0, 30, 0), (3.5, 0, -30, 0), (4.0, 0, 0, 0),
    (5.0, 0, 0, 90), (5.5, 0, 0, -90), (6.0, 0, 0, 0),
]


def _control(controller, wanted, state, dt):
    wanted_rates = torch.deg2rad(torch.tensor([wanted], dtype=torch.float32, device=state.pos.device))
    torque = controller.rate.update(wanted_rates, state.rates, dt)

    # Height hold (position and velocity layers on z only), so the drone does not sink to the ground during the test.
    hold_point = state.pos.clone()
    hold_point[:, 2] = TEST.start_z
    wanted_velocity = controller.position.update(hold_point, state.pos, dt) * torch.tensor([0.0, 0.0, 1.0], device=state.pos.device)
    vertical_acceleration = controller.velocity.update(wanted_velocity, state.vel, dt)[:, 2]
    thrust = controller.attitude.mass * (GRAVITY + vertical_acceleration)
    return controller.to_pwm(thrust, torque)


TEST = LayerTest(
    name="rate", title="Rate layer: body rate -> torque (height held)",
    labels=["Roll rate [deg/s]", "Pitch rate [deg/s]", "Yaw rate [deg/s]"],
    start_z=3.0, duration=7.0,
    setpoint=lambda t: [float(v) for v in step_table(STEPS, t)],
    control=_control,
    measure=lambda state: torch.rad2deg(state.rates[0]).tolist(),
)

if __name__ == "__main__":
    run(TEST)
