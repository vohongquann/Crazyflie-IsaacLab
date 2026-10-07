"""Attitude layer: wanted roll, pitch, yaw and thrust -> thrust and wanted body rates.

Test: attitude and rate layers fly. The setpoint is a roll, pitch and yaw angle; the position and velocity layers hold
the height (z only), and the thrust that keeps the vertical force is computed from the tilt.

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/attitude.py        # Isaac Sim window + live plot
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/attitude.py --viz none --no_live   # PNG only
"""
import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.flight import up_axis_from_angles
from Drone_RL.uav.pid_control.flight_test import LayerTest, euler_from_quat, run, step_table
from Drone_RL.uav.pid_control.kinematics import quat_to_rotmat
from Drone_RL.uav.pid_control.pid import PID

GRAVITY = 9.81


ATTITUDE_KP = (8.6, 8.6, 4.0)                # (roll, pitch, yaw), output [rad/s]
ATTITUDE_MAX_RATE = (6.0, 6.0, 3.0)          # [rad/s] limit of the output


class AttitudeController:
    def __init__(self, device, mass: float = U.DRONE_MASS_TOTAL_KG):
        self.mass = mass
        kp = torch.tensor(ATTITUDE_KP, device=device)
        out_limit = torch.tensor(ATTITUDE_MAX_RATE, device=device)
        self.pid = PID(kp=kp, out_limit=out_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, command: torch.Tensor, quat: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Command [roll, pitch, yaw (rad), thrust (N)] (N, 4) and orientation ``quat`` (x, y, z, w) -> thrust (N,) [N],
        wanted body rates [rad/s]. The thrust goes through (like the Crazyflie firmware).

        Roll, pitch and yaw are the angles of Rz(yaw) Ry(pitch) Rx(roll). The tilt error is the rotation that turns the
        body z axis onto the z axis of that orientation, seen from the body; the yaw error is wrapped to (-pi, pi].
        """
        rotation = quat_to_rotmat(quat)                                                   # (N, 3, 3), body -> world
        up_axis = rotation[:, :, 2]                                                       # z of the body in the world
        wanted_up_axis = up_axis_from_angles(command[:, 0], command[:, 1], command[:, 2])

        # Axis that turns z_body onto the wanted up axis, seen from the body: R^T (z_body x wanted).
        tilt_world = torch.linalg.cross(up_axis, wanted_up_axis)
        tilt_body = torch.einsum("nji,nj->ni", rotation, tilt_world)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        yaw_difference = command[:, 2] - yaw
        yaw_error = torch.atan2(torch.sin(yaw_difference), torch.cos(yaw_difference))     # in (-pi, pi]
        error = torch.stack([tilt_body[:, 0], tilt_body[:, 1], yaw_error], dim=-1)
        return command[:, 3], self.pid.update(error, dt)


# (t_start [s], roll, pitch, yaw) in deg
STEPS = [
    (0.0, 0, 0, 0),
    (1.0, 15, 0, 0), (2.5, 0, 0, 0),
    (3.5, 0, 15, 0), (5.0, 0, 0, 0),
    (6.0, -15, -15, 0), (7.5, 0, 0, 0),
    (8.5, 0, 0, 30), (10.0, 0, 0, 0),
]


def _control(controller, wanted, state, dt):
    roll, pitch, yaw = (torch.deg2rad(torch.tensor(v, device=state.pos.device)) for v in wanted)
    # Height hold (position and velocity layers on z only, as in rate.py): without it every tilt step loses some
    # height for good and the drone hit the ground halfway through the test. The thrust of the velocity layer is divided
    # by cos(roll) cos(pitch), so the vertical force stays the same at that tilt.
    hold_point = state.pos.clone()
    hold_point[:, 2] = TEST.start_z
    wanted_velocity = controller.position.update(hold_point, state.pos, dt) * torch.tensor([0.0, 0.0, 1.0], device=state.pos.device)
    hold_thrust = controller.velocity.update(torch.cat([wanted_velocity, torch.zeros_like(wanted_velocity[:, :1])], dim=-1), state.vel, dt)[:, 3]
    thrust = hold_thrust / (torch.cos(roll) * torch.cos(pitch))
    command = torch.stack([roll.expand_as(thrust), pitch.expand_as(thrust), yaw.expand_as(thrust), thrust], dim=-1)
    thrust, wanted_rates = controller.attitude.update(command, state.quat, dt)
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
