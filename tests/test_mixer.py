import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.mixer import force_allocation_inverse, wrench_matrix


def test_wrench_matrix_layout():
    """x configuration, L = 35.36 mm, motors m1..m4 at (+x,-y), (-x,-y), (-x,+y), (+x,+y), spin (-, +, -, +)."""
    length, km = U.CF_MOTOR_HALF_XY_M, U.DRONE_KM
    expected = torch.tensor([
        [1, 1, 1, 1],
        [-length, -length, length, length],
        [-length, length, length, -length],
        [-km, km, -km, km],
    ], dtype=torch.float64)
    assert torch.allclose(wrench_matrix(), expected)


def test_force_allocation_is_the_inverse_of_the_wrench_matrix():
    product = wrench_matrix().float() @ force_allocation_inverse()
    assert torch.allclose(product, torch.eye(4), atol=1e-5)


def test_force_allocation_reproduces_the_wrench():
    wrench = torch.tensor([0.34, 0.002, -0.001, 0.0003])
    forces = force_allocation_inverse() @ wrench
    assert torch.allclose(wrench_matrix().float() @ forces, wrench, atol=1e-5)
