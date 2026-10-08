"""Mixer of the Crazyflie 2.1 Brushless: force of each motor [N] <-> total thrust and body torques.

Same convention as ``Propulsion.step`` (forces -> wrench), with F_i the thrust of motor i, (x_i, y_i) its position and
s_i its spin sign (``uav_cfg.CF_MOTOR_SPIN``):

    T     =  sum_i F_i
    tau_x =  sum_i F_i y_i          a motor on the left (y > 0) pushes that side up: roll +
    tau_y = -sum_i F_i x_i          a motor in front (x > 0) pushes the nose up: pitch -
    tau_z =  km sum_i s_i F_i       reaction torque of the propeller drag, km = ``uav_cfg.DRONE_KM``

``force_allocation_inverse`` is the inverse: the force each motor must produce for a wrench [T, tau_x, tau_y, tau_z].
It is used wherever the rate loop is the PID, which works in newtons and newton metres: the classical PID cascade, the
kinematic flight, and the gain tasks (``GainCascadeAction``, the PID rate layer). Only a policy that writes the four motor commands itself (``MotorAction``, the landing
task) does not need it.
"""

import torch

from Drone_RL.uav import uav_cfg as U


def wrench_matrix(device: str = "cpu") -> torch.Tensor:
    """Matrix (4, 4): [F1 .. F4] [N] -> [T, tau_x, tau_y, tau_z]."""
    xy = torch.tensor(U.CF_MOTOR_XY, dtype=torch.float64, device=device)
    spin = torch.tensor(U.CF_MOTOR_SPIN, dtype=torch.float64, device=device)
    return torch.stack(
        [
            torch.ones(4, dtype=torch.float64, device=device),
            xy[:, 1],
            -xy[:, 0],
            U.DRONE_KM * spin,
        ]
    )


def force_allocation_inverse(device: str = "cpu") -> torch.Tensor:
    """Matrix (4, 4): wrench [T, tau_x, tau_y, tau_z] -> [F1 .. F4] [N] (float32)."""
    return torch.linalg.inv(wrench_matrix(device)).float()

