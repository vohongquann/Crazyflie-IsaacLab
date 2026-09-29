"""Position layer task (``Isaac-UAV-Position-RL-v0``): target position -> velocity, 50 Hz.

Frozen velocity, attitude and rate layers below; the command is the random target itself.
Shared scene, command, action, observation, events and terminations: ``cascade_env_cfg.py``. Rewards: ``mdp/rewards.py``.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.rl_control.cascade_env_cfg import LayerEnvCfg, termination_penalty


@configclass
class PositionRewardsCfg:
    position_coarse = RewTerm(func=mdp.position_tracking, weight=2.0, params={"std": 1.0})
    position_error = RewTerm(func=mdp.position_error_l2, weight=-0.1)
    position_fine = RewTerm(func=mdp.position_tracking, weight=1.0, params={"std": 0.1})
    output_change = RewTerm(func=mdp.output_change, weight=-0.05)
    terminated = termination_penalty(-500.0)


@configclass
class PositionEnvCfg(LayerEnvCfg):
    LAYER = "position"
    EPISODE_S = 8.0
    rewards: PositionRewardsCfg = PositionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.layer.resampling_time_range = (3.0, 5.0)
