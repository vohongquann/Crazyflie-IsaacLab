# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Crazyflie constants."""

import Drone_RL.tasks  # noqa: F401
from Drone_RL.uav import uav_cfg


def test_crazyflie_constants_are_consistent():
    """The derived thrust model reproduces the nominal weight at the hover throttle."""
    a, b, c = uav_cfg.CF_THRUST_COEF_G
    pwm = uav_cfg.DRONE_HOVER_THROTTLE * uav_cfg.CF_PWM_MAX
    total_grams = a * pwm**2 + b * pwm + c
    assert abs(total_grams / 1000.0 - uav_cfg.DRONE_MASS_TOTAL_KG) < 1e-4


def test_landing_task_is_registered():
    import gymnasium as gym

    assert "Isaac-UAV-Landing-ArUco-v0" in gym.registry
