"""Velocity layer task (``Isaac-UAV-Velocity-RL-v0``): velocity -> acceleration, 100 Hz.

Frozen attitude and rate layers below; the command comes from the PID position layer. Everything else is
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
class VelocityPolicyCfg(PolicyCfg):
    """Velocity error, velocity, thrust direction, last 2 outputs (15)."""

    velocity_error = ObsTerm(func=mdp.velocity_error)
    velocity = ObsTerm(func=isaac_mdp.root_lin_vel_w)
    body_up_in_world = ObsTerm(func=mdp.body_up_in_world)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class VelocityRewardsCfg(RewardsCfg):
    velocity_error = RewTerm(
        func=mdp.velocity_error_l2,
        weight=-0.1,
    )


@configclass
class VelocityEnvCfg(LayerEnvCfg):
    LAYER = "velocity"
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(offset=(0.5, 0.5, 0.5, 0.0)),     # [m/s]
    )
    observations: ObservationsCfg = ObservationsCfg(policy=VelocityPolicyCfg())
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()
