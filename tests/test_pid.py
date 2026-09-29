"""Plain PID: proportional, integral and derivative terms and the limits."""
import torch

from Drone_RL.uav.pid_control.pid import PID


def test_proportional():
    assert torch.allclose(PID(kp=2.0).update(torch.tensor([1.0, -3.0]), dt=0.1), torch.tensor([2.0, -6.0]))


def test_integral_accumulates_and_reset_clears():
    pid = PID(kp=0.0, ki=1.0)
    e = torch.tensor([1.0])
    pid.update(e, dt=0.5)
    assert torch.allclose(pid.update(e, dt=0.5), torch.tensor([1.0]))     # integral of 1 over 1 s
    pid.reset()
    assert torch.allclose(pid.update(e, dt=0.5), torch.tensor([0.5]))


def test_derivative_has_no_kick_on_first_call():
    pid = PID(kp=0.0, kd=1.0)
    assert torch.allclose(pid.update(torch.tensor([5.0]), dt=0.1), torch.tensor([0.0]))
    assert torch.allclose(pid.update(torch.tensor([5.2]), dt=0.1), torch.tensor([2.0]))   # (5.2 - 5.0) / 0.1


def test_limits():
    pid = PID(kp=10.0, ki=100.0, out_limit=1.0, int_limit=0.5)
    out = pid.update(torch.tensor([1.0]), dt=1.0)
    assert out.item() == 1.0 and pid.integral.item() == 0.5
