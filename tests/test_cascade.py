import math

import pytest
import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.pid_control.cascade import CascadePID
from Drone_RL.uav.pid_control.velocity import VelocityController

LEVEL = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
POS = torch.tensor([[0.0, 0.0, 0.5]])


def test_hover_command_is_the_hover_throttle():
    """At rest exactly on the target, all four motors get the hover PWM."""
    pwm = CascadePID("cpu").step(POS, POS, torch.zeros(1, 3), LEVEL, torch.zeros(1, 3), 0.002)
    assert torch.allclose(pwm, torch.full((1, 4), U.DRONE_HOVER_THROTTLE * U.CF_PWM_MAX), rtol=1e-3)


def test_target_ahead_pitches_the_nose_down():
    """A target in +x needs the front motors (m1, m4) to push less than the rear ones (m2, m3)."""
    pwm = CascadePID("cpu").step(POS + torch.tensor([[1.0, 0.0, 0.0]]), POS, torch.zeros(1, 3), LEVEL,
                                 torch.zeros(1, 3), 0.002)
    assert pwm[0, [0, 3]].mean() < pwm[0, [1, 2]].mean()


def test_layers_can_be_called_alone():
    """Entering at the velocity layer with zero wanted velocity gives the hover PWM too."""
    controller = CascadePID("cpu")
    command = controller.velocity.update(torch.zeros(1, 4), torch.zeros(1, 3), 0.002)
    thrust, rates = controller.attitude.update(command, LEVEL, 0.002)
    pwm = controller.to_pwm(thrust, controller.rate.update(rates, torch.zeros(1, 3), 0.002))
    assert torch.allclose(pwm, torch.full((1, 4), U.DRONE_HOVER_THROTTLE * U.CF_PWM_MAX), rtol=1e-3)


def test_velocity_layer_writes_roll_pitch_and_thrust():
    """No error: level, hover thrust. A velocity wanted ahead (+x) pitches nose-down, to the left (+y) rolls left, up
    raises the thrust; the horizontal error is turned into the yaw frame; the output is limited."""
    velocity = CascadePID("cpu").velocity
    weight = U.DRONE_MASS_TOTAL_KG * 9.81
    out = velocity.update(torch.zeros(1, 4), torch.zeros(1, 3), 0.01)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 0.0, weight]]), atol=1e-6)
    out = VelocityController("cpu").update(torch.tensor([[0.5, 0.0, 0.0, 0.0]]), torch.zeros(1, 3), 0.01)
    assert out[0, 1] > 0 and abs(out[0, 0]) < 1e-6 and abs(out[0, 3] - weight) < 1e-6      # pitch +: nose down (rate layer sign)
    out = VelocityController("cpu").update(torch.tensor([[0.0, 0.5, 0.5, 0.0]]), torch.zeros(1, 3), 0.01)
    assert out[0, 0] < 0 and abs(out[0, 1]) < 1e-6 and out[0, 3] > weight
    # wanted yaw 90 deg: the world +y direction is "ahead" for the drone
    out = VelocityController("cpu").update(torch.tensor([[0.0, 0.5, 0.0, math.pi / 2]]), torch.zeros(1, 3), 0.01)
    assert out[0, 1] > 0 and abs(out[0, 0]) < 1e-6 and out[0, 2] == pytest.approx(math.pi / 2)
    out = VelocityController("cpu").update(torch.tensor([[50.0, -50.0, 50.0, 0.0]]), torch.zeros(1, 3), 0.01)
    assert torch.allclose(out[0, [0, 1]].abs(), torch.tensor([0.6, 0.6])) and out[0, 3] == pytest.approx(weight + 0.27)


def test_attitude_layer_takes_angles_and_passes_the_thrust_through():
    """On the wanted angles the rates are zero; a roll error asks for a roll rate of the same sign; thrust is the command's."""
    attitude = CascadePID("cpu").attitude
    thrust = torch.tensor([0.37])
    level = torch.tensor([[0.0, 0.0, 0.0, 0.37]])
    out_thrust, rates = attitude.update(level, LEVEL, 0.004)
    assert torch.allclose(rates, torch.zeros(1, 3), atol=1e-6) and torch.equal(out_thrust, thrust)
    out_thrust, rates = attitude.update(torch.tensor([[0.2, -0.1, 0.0, 0.37]]), LEVEL, 0.004)
    assert rates[0, 0] > 0 and rates[0, 1] < 0 and torch.equal(out_thrust, thrust)
    _, rates = attitude.update(torch.tensor([[0.0, 0.0, 0.5, 0.37]]), LEVEL, 0.004)
    assert rates[0, 2] > 0 and torch.allclose(rates[0, :2], torch.zeros(2), atol=1e-6)
