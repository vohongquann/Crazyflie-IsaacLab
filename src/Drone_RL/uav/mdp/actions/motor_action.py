"""Action of the policy: four motor commands in [0, 1] -> thrust and torque applied to PhysX.

    a (N, 4) in [0, 1]  ->  PWM = a * 65535  ->  thrust curve  ->  motor lag  ->  F_i
                        ->  total thrust and body torques  ->  wrench on the body

The chain after the PWM is ``propulsion.Propulsion`` (formulas in guide/02_propulsion.md). There is no
controller in between: the network drives the motors directly, and any PID is added outside this term.
"""
from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .propulsion import SPIN_VISUAL_SCALE, Propulsion
from .constants import MOTOR_TAU_INC_RANGE, MOTOR_TAU_DEC_RANGE
from Drone_RL.uav.uav_cfg import (
    CF_DRAG_COEF,
    CF_F_MAX_N,
    CF_MOTOR_SPIN,
    CF_MOTOR_XY,
    CF_PWM_MAX,
    CF_THRUST_COEF_G,
    DRONE_HOVER_THRUST_N,
    DRONE_KM,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MotorAction(ActionTerm):
    """Policy output ``a`` in [0, 1]^4, one value per motor (order m1..m4 of ``cf2x.usd``)."""

    cfg: "MotorActionCfg"

    def __init__(self, cfg: "MotorActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._body_id = self._robot.find_bodies(cfg.body_name)[0]
        self._phys_dt = env.physics_dt

        self._pwm_max = cfg.pwm_max
        self._propulsion = Propulsion(cfg, self.num_envs, self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._pwm = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._pwm

    def reset(self, env_ids):
        self._raw_actions[env_ids] = 0.0
        self._pwm[env_ids] = 0.0
        self._propulsion.reset(env_ids)
        # Start every episode with the motors at hover thrust, as if the drone had been flying (as CascadeAction).
        self._propulsion.force[env_ids] = DRONE_HOVER_THRUST_N / 4.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = (self.cfg.offset + self.cfg.scale * actions).clamp(0.0, 1.0)
        self._pwm[:] = self._raw_actions * self._pwm_max

    def apply_actions(self):
        self._propulsion.step(self._pwm, self._robot, self._body_id, self._phys_dt)


@configclass
class MotorActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MotorAction

    asset_name: str = MISSING
    offset: float = 0.0
    scale: float = 1.0   # motor command = clamp(offset + scale * policy output, 0, 1); centre it on the hover throttle for RL
    body_name: str = "body"

    thrust_coef: tuple[float, float, float] = CF_THRUST_COEF_G
    f_max: float = CF_F_MAX_N
    pwm_max: float = CF_PWM_MAX
    gravity: float = 9.81
    km: float = DRONE_KM   # yaw torque per unit of thrust
    motor_xy: tuple = CF_MOTOR_XY
    motor_spin: tuple[float, float, float, float] = CF_MOTOR_SPIN

    use_motor_lag: bool = True
    tau_inc_range: tuple[float, float] = MOTOR_TAU_INC_RANGE
    tau_dec_range: tuple[float, float] = MOTOR_TAU_DEC_RANGE
    use_air_drag: bool = True
    drag_coef: float = CF_DRAG_COEF
    drag_scale: tuple[float, float] = (0.5, 1.5)
    use_thrust_noise: bool = False
    thrust_noise_std: float = 0.01
    use_motor_asymmetry: bool = True
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    spin_propellers: bool = True   # turn the propeller joints with the rotor speed (visual only)
    spin_visual_scale: float = SPIN_VISUAL_SCALE   # picture speed / rotor speed
