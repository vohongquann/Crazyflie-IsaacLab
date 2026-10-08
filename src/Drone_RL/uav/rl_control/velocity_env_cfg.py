"""Velocity layer task: velocity -> roll, pitch, thrust, 100 Hz (base of ``Isaac-UAV-Velocity-Gains-v0``).

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
    """Velocity error, velocity, thrust direction, last outputs (the gains)."""

    velocity_error = ObsTerm(func=mdp.velocity_error)
    velocity = ObsTerm(func=isaac_mdp.root_lin_vel_w)
    body_up_in_world = ObsTerm(func=mdp.body_up_in_world)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class VelocityRewardsCfg(RewardsCfg):
    # One term per axis (x, y, z): the weights sum to 1 for the coarse and for the fine terms, as before.
    velocity_x_coarse = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 0},
    )
    velocity_x_fine = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 0},
    )
    velocity_y_coarse = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 1},
    )
    velocity_y_fine = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 1},
    )
    velocity_z_coarse = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 2},
    )
    velocity_z_fine = RewTerm(
        func=mdp.velocity_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 2},
    )
    # Penalties 10 times, 5 times and 5 times heavier than in the first run (-0.1, -1, -0.02): that run stopped at
    # 81-84 % of the wanted velocity, with no overshoot and almost no ringing.
    velocity_error = RewTerm(
        func=mdp.velocity_error_l2,
        weight=-1.0,
    )
    velocity_overshoot = RewTerm(                     # step response: overshoot
        func=mdp.velocity_overshoot,
        weight=-5.0,
    )
    velocity_ringing = RewTerm(                       # step response: oscillation once settled
        func=mdp.velocity_ringing,
        weight=-0.1,
        params={"std": 0.25},
    )


@configclass
class VelocityEnvCfg(LayerEnvCfg):
    LAYER = "velocity"
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(offset=(0.5, 0.5, 0.5, 0.0)),     # [m/s]
    )
    observations: ObservationsCfg = ObservationsCfg(policy=VelocityPolicyCfg())
    rewards: VelocityRewardsCfg = VelocityRewardsCfg()
