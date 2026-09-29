"""Velocity layer: velocity error -> wanted acceleration.

Test: velocity, attitude and rate layers fly. The setpoint steps the world velocity (vx, vy, vz).

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/velocity.py --live --viz kit
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/velocity.py        # no window, PNG only
"""
import torch

from Drone_RL.uav.pid_control.flight_test import LayerTest, run, step_table

from Drone_RL.uav.pid_control.pid import PID


class VelocityController:
    def __init__(self, device):
        kp = torch.tensor([9.3, 9.3, 20.0], device=device)
        ki = torch.tensor([0.9, 0.9, 1.9], device=device)
        int_limit = torch.tensor([2.0, 2.0, 1.0], device=device)
        out_limit = torch.tensor([7.0, 7.0, 6.0], device=device)     # [m/s^2]
        self.pid = PID(kp=kp, ki=ki, int_limit=int_limit, out_limit=out_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, wanted_velocity: torch.Tensor, vel: torch.Tensor, dt: float) -> torch.Tensor:
        """Wanted velocity and velocity (world, (N, 3) [m/s]) -> wanted acceleration [m/s^2]."""
        return self.pid.update(wanted_velocity - vel, dt)


# (t_start [s], vx, vy, vz) in m/s
STEPS = [
    (0.0, 0, 0, 0),
    (1.0, 1.0, 0, 0), (3.0, 0, 0, 0),
    (4.5, 0, 1.0, 0), (6.5, 0, 0, 0),
    (8.0, 0, 0, 0.5), (9.5, 0, 0, 0),
    (11.0, -1.0, -1.0, 0), (13.0, 0, 0, 0),
]


def _control(controller, wanted, state, dt):
    wanted_velocity = torch.tensor([wanted], dtype=torch.float32, device=state.pos.device)
    acceleration = controller.velocity.update(wanted_velocity, state.vel, dt)
    thrust, wanted_rates = controller.attitude.update(acceleration, state.quat, dt)
    return controller.to_pwm(thrust, controller.rate.update(wanted_rates, state.rates, dt))


TEST = LayerTest(
    name="velocity", title="Velocity layer: velocity -> acceleration",
    labels=["Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    start_z=2.0, duration=14.5,
    setpoint=lambda t: [float(v) for v in step_table(STEPS, t)],
    control=_control,
    measure=lambda state: state.vel[0].tolist(),
)

if __name__ == "__main__":
    run(TEST)
