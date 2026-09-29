import torch

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.pid_control.cascade import CascadePID

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
    acceleration = controller.velocity.update(torch.zeros(1, 3), torch.zeros(1, 3), 0.002)
    thrust, rates = controller.attitude.update(acceleration, LEVEL, 0.002)
    pwm = controller.to_pwm(thrust, controller.rate.update(rates, torch.zeros(1, 3), 0.002))
    assert torch.allclose(pwm, torch.full((1, 4), U.DRONE_HOVER_THROTTLE * U.CF_PWM_MAX), rtol=1e-3)
