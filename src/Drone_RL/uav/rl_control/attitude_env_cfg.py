"""Attitude layer task (``Isaac-UAV-Attitude-RL-v0``): acceleration and yaw -> body rates and thrust, 50 Hz.

Frozen rate layer below; the command comes from the PID position and velocity layers.
Shared scene, command, action, observation, events and terminations: ``cascade_env_cfg.py``. Rewards: ``mdp/rewards.py``.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.rl_control.cascade_env_cfg import LayerEnvCfg, termination_penalty


@configclass
class AttitudeRewardsCfg:
    tilt = RewTerm(func=mdp.tilt_tracking, weight=2.0, params={"std": 0.2})
    tilt_error = RewTerm(func=mdp.tilt_error_l2, weight=-1.0)
    yaw = RewTerm(func=mdp.yaw_tracking, weight=1.0, params={"std": 0.4})
    acceleration = RewTerm(func=mdp.acceleration_tracking, weight=1.0, params={"std": 2.0})
    output_change = RewTerm(func=mdp.output_change, weight=-0.05)
    spin = RewTerm(func=mdp.body_rates_l2, weight=-0.01)
    terminated = termination_penalty(-500.0)


@configclass
class AttitudeEnvCfg(LayerEnvCfg):
    LAYER = "attitude"
    rewards: AttitudeRewardsCfg = AttitudeRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.layer.offset = (2.0, 2.0, 1.5, 0.0)          # [m/s^2]
