"""Plain PID controller, batched over environments (torch tensors of any shape)."""
from __future__ import annotations

import torch


class PID:
    """out = kp * e + ki * integral(e dt) + kd * de/dt

    The integral is clamped to ``+-int_limit`` (anti-windup) and the output to ``+-out_limit``; ``None`` means no limit.
    Gains and limits are floats or tensors that broadcast with the error.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, out_limit=None, int_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.int_limit = out_limit, int_limit
        self.integral = None       # integral of the error, same shape as the error
        self.prev_error = None     # error of the previous call, for the derivative

    def reset(self, env_ids=None) -> None:
        """Clear the state of all environments, or only of the rows ``env_ids``."""
        if self.integral is None:
            return
        if env_ids is None:
            self.integral, self.prev_error = None, None
        else:
            self.integral[env_ids] = 0.0
            self.prev_error[env_ids] = 0.0

    def update(self, error: torch.Tensor, dt: float) -> torch.Tensor:
        """One step with the error ``setpoint - measurement`` and the time since the last call ``dt`` [s]."""
        if self.integral is None:
            self.integral = torch.zeros_like(error)
            self.prev_error = error.clone()      # first call: no derivative kick

        self.integral = self.integral + error * dt                      # I: running integral of e
        if self.int_limit is not None:
            self.integral = self.integral.clamp(-self.int_limit, self.int_limit)
        derivative = (error - self.prev_error) / dt                     # D: slope of e
        self.prev_error = error.clone()

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_limit is not None:
            out = out.clamp(-self.out_limit, self.out_limit)
        return out
