"""Position layer task (``Isaac-UAV-Position-RL-v0``): target position -> velocity, 50 Hz.

Frozen velocity, attitude and rate layers below; the command is the random target itself. Everything else is
``cascade_env_cfg.py``.
"""
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.mdp.commands import LayerCommandCfg
from Drone_RL.uav.rl_control.cascade_env_cfg import (
    CommandsCfg,
    LayerEnvCfg,
    ObservationsCfg,
    PolicyCfg,
    RewardsCfg,
)


@configclass
class PositionPolicyCfg(PolicyCfg):
    """Position error, velocity, last 2 outputs (12)."""

    position_error = ObsTerm(func=mdp.position_error)
    velocity = ObsTerm(func=isaac_mdp.root_lin_vel_w)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class PositionRewardsCfg(RewardsCfg):
    position_error = RewTerm(
        func=mdp.position_error_l2,
        weight=-0.1,
    )


@configclass
class PositionEnvCfg(LayerEnvCfg):
    LAYER = "position"
    EPISODE_S = 8.0
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(resampling_time_range=(3.0, 5.0)),
    )
    observations: ObservationsCfg = ObservationsCfg(policy=PositionPolicyCfg())
    rewards: PositionRewardsCfg = PositionRewardsCfg()
