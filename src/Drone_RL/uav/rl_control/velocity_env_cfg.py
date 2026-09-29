"""Velocity layer task (``Isaac-UAV-Velocity-RL-v0``): velocity -> acceleration, 50 Hz.

Frozen attitude and rate layers below; the command comes from the PID position layer.
Shared scene, command, action, observation, events and terminations: ``cascade_env_cfg.py``. Rewards: ``mdp/rewards.py``.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.rl_control.cascade_env_cfg import LayerEnvCfg, termination_penalty


@configclass
class VelocityRewardsCfg:
    velocity_coarse = RewTerm(func=mdp.velocity_tracking, weight=2.0, params={"std": 1.0})
    velocity_error = RewTerm(func=mdp.velocity_error_l2, weight=-0.1)
    velocity_fine = RewTerm(func=mdp.velocity_tracking, weight=1.0, params={"std": 0.15})
    output_change = RewTerm(func=mdp.output_change, weight=-0.05)
    terminated = termination_penalty(-500.0)


@configclass
class VelocityEnvCfg(LayerEnvCfg):
    LAYER = "velocity"
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.layer.offset = (0.5, 0.5, 0.5, 0.0)          # [m/s]
