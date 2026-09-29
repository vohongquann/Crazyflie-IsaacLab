"""Attitude layer: wanted acceleration -> thrust and wanted body rates.

Test: attitude and rate layers fly. The setpoint is a roll, pitch and yaw angle; ``angles_to_acceleration`` turns the
tilt into the horizontal acceleration that holds it (vertical component kept at g, so the height stays about constant).

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/attitude.py --live --viz kit
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/attitude.py        # no window, PNG only
"""
import math

import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.pid_control.flight_test import LayerTest, euler_from_quat, run, step_table
from Drone_RL.uav.pid_control.kinematics import quat_to_rotmat
from Drone_RL.uav.pid_control.pid import PID

GRAVITY = 9.81


class AttitudeController:
    def __init__(self, device, mass: float = U.DRONE_MASS_TOTAL_KG):
        self.mass = mass
        self.gravity = torch.tensor([0.0, 0.0, GRAVITY], device=device)
        kp = torch.tensor([17.3, 17.3, 4.0], device=device)             # (roll, pitch, yaw)
        out_limit = torch.tensor([6.0, 6.0, 3.0], device=device)      # [rad/s]
        self.pid = PID(kp=kp, out_limit=out_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, wanted_acceleration: torch.Tensor, quat: torch.Tensor, dt: float, wanted_yaw: float = 0.0
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wanted acceleration (world) and orientation ``quat`` (x, y, z, w) -> thrust (N,) [N], wanted body rates [rad/s].

        The motors must push with force per unit mass f = a + g e_z, along f / |f|. The yaw turns to
        ``wanted_yaw`` [rad] (0 by default).
        """
        rotation = quat_to_rotmat(quat)                                                   # (N, 3, 3), body -> world
        up_axis = rotation[:, :, 2]                                                       # z of the body in the world
        force_per_mass = wanted_acceleration + self.gravity
        wanted_up_axis = force_per_mass / force_per_mass.norm(dim=-1, keepdim=True)
        thrust = self.mass * (force_per_mass * up_axis).sum(-1).clamp(min=0.0)            # T = m f . z_body

        # Axis that turns z_body onto the wanted up axis, seen from the body: R^T (z_body x wanted).
        tilt_world = torch.linalg.cross(up_axis, wanted_up_axis)
        tilt_body = torch.einsum("nji,nj->ni", rotation, tilt_world)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        yaw_difference = wanted_yaw - yaw
        yaw_error = torch.atan2(torch.sin(yaw_difference), torch.cos(yaw_difference))     # in (-pi, pi]
        error = torch.stack([tilt_body[:, 0], tilt_body[:, 1], yaw_error], dim=-1)
        return thrust, self.pid.update(error, dt)


def angles_to_acceleration(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Roll, pitch, yaw [rad] (order Rz Ry Rx) -> horizontal acceleration [m/s^2] (1, 3) that holds that tilt."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    z_axis = (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)       # third column of the rotation matrix
    return torch.tensor([[GRAVITY * z_axis[0] / z_axis[2], GRAVITY * z_axis[1] / z_axis[2], 0.0]])


# (t_start [s], roll, pitch, yaw) in deg
STEPS = [
    (0.0, 0, 0, 0),
    (1.0, 15, 0, 0), (2.5, 0, 0, 0),
    (3.5, 0, 15, 0), (5.0, 0, 0, 0),
    (6.0, -15, -15, 0), (7.5, 0, 0, 0),
    (8.5, 0, 0, 30), (10.0, 0, 0, 0),
]


def _control(controller, wanted, state, dt):
    roll, pitch, yaw = (math.radians(v) for v in wanted)
    acceleration = angles_to_acceleration(roll, pitch, yaw).to(state.pos.device)
    thrust, wanted_rates = controller.attitude.update(acceleration, state.quat, dt, wanted_yaw=yaw)
    return controller.to_pwm(thrust, controller.rate.update(wanted_rates, state.rates, dt))


TEST = LayerTest(
    name="attitude", title="Attitude layer: roll, pitch, yaw angle -> body rate",
    labels=["Roll [deg]", "Pitch [deg]", "Yaw [deg]"],
    start_z=3.0, duration=11.5,
    setpoint=lambda t: [float(v) for v in step_table(STEPS, t)],
    control=_control,
    measure=lambda state: torch.rad2deg(euler_from_quat(state.quat)[0]).tolist(),
)

if __name__ == "__main__":
    run(TEST)
