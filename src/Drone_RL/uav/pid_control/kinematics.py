"""Known trajectories and differential flatness for the quadrotor (pure torch, no Isaac dependency).

A quadrotor is differentially flat in (x, y, z, yaw): position, velocity, attitude, body rates and the
wrench that would produce the motion all follow from the trajectory and its derivatives. That lets the
drone be driven kinematically (write the pose from the trajectory, no physics) while still reporting
whether the real motors could fly it. Formulas: guide/03_kinematic_flight.md. Quaternions are (x, y, z, w).
"""
from __future__ import annotations

import math

import torch

GRAVITY = 9.81
DIFF_STEP = 2e-3          # h, step of the finite differences [s]


# ── 4.1 Trajectory ──────────────────────────────────────────────────────────────────────

def _min_jerk_ramp(tau: float) -> float:
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: 0 to 1 with zero velocity and acceleration at both ends."""
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def trajectory(pattern: str, t: float, hover_z: float = 0.5, size: float = 0.5, period: float = 8.0,
               takeoff_time: float = 2.0, start_z: float = 0.1) -> tuple[torch.Tensor, float]:
    """Position (x, y, z) [m] and yaw [rad] at time ``t`` [s]: takeoff on a ramp, then the pattern."""
    z = start_z + (hover_z - start_z) * _min_jerk_ramp(t / takeoff_time)

    t_pattern = max(t - takeoff_time, 0.0)                 # t_p
    blend = _min_jerk_ramp(t_pattern / 2.0)                # b: eases the pattern in so the velocity starts at zero
    omega = 2 * math.pi / period
    if pattern == "hover":
        x = y = 0.0
    elif pattern == "circle":
        x = size * (math.cos(omega * t_pattern) - 1.0) * blend
        y = size * math.sin(omega * t_pattern) * blend
    elif pattern == "figure8":
        x = size * math.sin(omega * t_pattern) * blend
        y = 0.5 * size * math.sin(2 * omega * t_pattern) * blend
    else:
        raise ValueError(pattern)
    return torch.tensor([x, y, z], dtype=torch.float64), 0.0


# ── 4.2 Velocity and acceleration ───────────────────────────────────────────────────────

def _position_velocity_acceleration(pattern: str, t: float, **traj_kw):
    """Position, velocity, acceleration (central differences) and yaw at time ``t``."""
    h = DIFF_STEP
    before, _ = trajectory(pattern, t - h, **traj_kw)
    position, yaw = trajectory(pattern, t, **traj_kw)
    after, _ = trajectory(pattern, t + h, **traj_kw)
    velocity = (after - before) / (2 * h)
    acceleration = (after - 2 * position + before) / h**2
    return position, velocity, acceleration, yaw


# ── 4.3 Differential flatness ───────────────────────────────────────────────────────────

def _attitude(acceleration: torch.Tensor, yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation matrix R (body -> world) and force per unit mass f, from steps (1)-(4)."""
    force_per_mass = acceleration + torch.tensor([0.0, 0.0, GRAVITY], dtype=torch.float64)    # (1) f = a + g e_z
    z_body = force_per_mass / force_per_mass.norm()                                           # (2) up axis
    x_heading = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], dtype=torch.float64)       # (3) x_c
    y_body = torch.linalg.cross(z_body, x_heading)                                            # (4) y_b
    y_body = y_body / y_body.norm()
    x_body = torch.linalg.cross(y_body, z_body)
    return torch.stack([x_body, y_body, z_body], dim=1), force_per_mass


def _rotation_at(pattern: str, t: float, **traj_kw) -> torch.Tensor:
    _, _, acceleration, yaw = _position_velocity_acceleration(pattern, t, **traj_kw)
    return _attitude(acceleration, yaw)[0]


def _body_rates_at(pattern: str, t: float, **traj_kw) -> torch.Tensor:
    """Step (5): omega_b = vee(R^T dR/dt), dR/dt by central difference of R."""
    h = DIFF_STEP
    rotation_rate = (_rotation_at(pattern, t + h, **traj_kw) - _rotation_at(pattern, t - h, **traj_kw)) / (2 * h)
    skew = _rotation_at(pattern, t, **traj_kw).T @ rotation_rate                 # skew-symmetric matrix
    return torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])                     # vee


def state(pattern: str, t: float, mass: float, inertia: tuple[float, float, float], **traj_kw) -> dict:
    """Everything at time ``t``: position, velocity, orientation, body rates, and the thrust [N] and torque [N m] needed."""
    position, velocity, acceleration, yaw = _position_velocity_acceleration(pattern, t, **traj_kw)
    rotation, force_per_mass = _attitude(acceleration, yaw)

    body_rates = _body_rates_at(pattern, t, **traj_kw)                                            # (5)
    h = DIFF_STEP
    angular_acceleration = (_body_rates_at(pattern, t + 4 * h, **traj_kw)
                            - _body_rates_at(pattern, t - 4 * h, **traj_kw)) / (8 * h)            # (6)

    J = torch.tensor(inertia, dtype=torch.float64)
    return {
        "position": position,
        "velocity": velocity,
        "quat": rotmat_to_quat(rotation),
        "body_rates": body_rates,
        "thrust": mass * (force_per_mass * rotation[:, 2]).sum(),                                 # (2) T = m |f|
        "torque": J * angular_acceleration + torch.linalg.cross(body_rates, J * body_rates),      # (7) Euler
    }


# ── Rotation conversions ────────────────────────────────────────────────────────────────

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) quaternion (x, y, z, w) -> (N,3,3) rotation matrix, body -> world."""
    x, y, z, w = q.unbind(-1)
    r = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return r.reshape(-1, 3, 3)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(3,3) rotation matrix -> quaternion (x, y, z, w)."""
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        return torch.stack([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4])
    i = int(torch.argmax(torch.diagonal(R)))          # the largest diagonal entry keeps the division well conditioned
    j, k = (i + 1) % 3, (i + 2) % 3
    s = torch.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = torch.zeros(4, dtype=R.dtype)
    q[i], q[3] = s / 4, (R[k, j] - R[j, k]) / s
    q[j], q[k] = (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return q
