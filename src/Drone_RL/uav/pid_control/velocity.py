"""Velocity layer: velocity error -> wanted roll, pitch and thrust (like the velocity controller of the Crazyflie firmware).

Test: velocity, attitude and rate layers fly. The setpoint steps the world velocity (vx, vy, vz).

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/velocity.py        # Isaac Sim window + live plot
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/velocity.py --viz none --no_live   # PNG only
"""
import torch

from Drone_RL.uav.mdp.flight import WEIGHT_N
from Drone_RL.uav.pid_control.flight_test import LayerTest, run, step_table

from Drone_RL.uav.pid_control.pid import PID


# One value per output, none shared. Axes of the yaw frame (x ahead, y to the left, z up): x gives the pitch, y the roll, z the
# thrust. Units: pitch and roll [rad per m/s], thrust [N per m/s].
VELOCITY_KP_PITCH = 0.95
VELOCITY_KP_ROLL = 0.95
VELOCITY_KP_THRUST = 0.9
VELOCITY_KI_PITCH = 0.092
VELOCITY_KI_ROLL = 0.092
VELOCITY_KI_THRUST = 0.086
VELOCITY_INT_LIMIT_PITCH = 2.0               # limit of the integral of the error [m] (anti-windup)
VELOCITY_INT_LIMIT_ROLL = 2.0
VELOCITY_INT_LIMIT_THRUST = 1.0
VELOCITY_MAX_PITCH = 0.6                     # limit of the output: pitch and roll [rad], thrust above or below hover [N]
VELOCITY_MAX_ROLL = 0.6
VELOCITY_MAX_THRUST = 0.27

# (x, y, z) of the PID: (pitch, roll, thrust)
VELOCITY_KP = (VELOCITY_KP_PITCH, VELOCITY_KP_ROLL, VELOCITY_KP_THRUST)
VELOCITY_KI = (VELOCITY_KI_PITCH, VELOCITY_KI_ROLL, VELOCITY_KI_THRUST)
VELOCITY_INT_LIMIT = (VELOCITY_INT_LIMIT_PITCH, VELOCITY_INT_LIMIT_ROLL, VELOCITY_INT_LIMIT_THRUST)
VELOCITY_MAX = (VELOCITY_MAX_PITCH, VELOCITY_MAX_ROLL, VELOCITY_MAX_THRUST)


class VelocityController:
    def __init__(self, device):
        kp = torch.tensor(VELOCITY_KP, device=device)
        ki = torch.tensor(VELOCITY_KI, device=device)
        int_limit = torch.tensor(VELOCITY_INT_LIMIT, device=device)
        out_limit = torch.tensor(VELOCITY_MAX, device=device)
        self.pid = PID(kp=kp, ki=ki, int_limit=int_limit, out_limit=out_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, command: torch.Tensor, vel: torch.Tensor, dt: float) -> torch.Tensor:
        """Command [wanted velocity (world, m/s), wanted yaw (rad)] (N, 4) and velocity (world, (N, 3) [m/s]) -> attitude
        command [roll, pitch, yaw, thrust (N)] (N, 4).

        The velocity error is turned into the yaw frame (x ahead, y to the left); ahead needs a nose-down pitch, to the
        left a roll to the left (+roll lifts the left side, like the rate and attitude layers). The thrust is the
        weight plus the output of the z axis.
        """
        error = command[:, :3] - vel
        cos, sin = torch.cos(command[:, 3]), torch.sin(command[:, 3])
        error_in_yaw_frame = torch.stack(
            [cos * error[:, 0] + sin * error[:, 1], -sin * error[:, 0] + cos * error[:, 1], error[:, 2]], dim=-1)
        out = self.pid.update(error_in_yaw_frame, dt)
        return torch.stack([-out[:, 1], out[:, 0], command[:, 3], WEIGHT_N + out[:, 2]], dim=-1)


# (t_start [s], vx, vy, vz) in m/s
STEPS = [
    (0.0, 0, 0, 0),
    (1.0, 1.0, 0, 0), (3.0, 0, 0, 0),
    (4.5, 0, 1.0, 0), (6.5, 0, 0, 0),
    (8.0, 0, 0, 0.5), (9.5, 0, 0, 0),
    (11.0, -1.0, -1.0, 0), (13.0, 0, 0, 0),
]


def _control(controller, wanted, state, dt):
    command = torch.tensor([[*wanted, 0.0]], dtype=torch.float32, device=state.pos.device)      # wanted yaw 0
    command = controller.velocity.update(command, state.vel, dt)
    thrust, wanted_rates = controller.attitude.update(command, state.quat, dt)
    return controller.to_pwm(thrust, controller.rate.update(wanted_rates, state.rates, dt))


TEST = LayerTest(
    name="velocity", title="Velocity layer: velocity -> roll, pitch, thrust",
    labels=["Vx [m/s]", "Vy [m/s]", "Vz [m/s]"],
    start_z=2.0, duration=14.5,
    setpoint=lambda t: [float(v) for v in step_table(STEPS, t)],
    control=_control,
    measure=lambda state: state.vel[0].tolist(),
)

if __name__ == "__main__":
    run(TEST)
