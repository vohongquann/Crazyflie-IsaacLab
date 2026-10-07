"""Position layer task (``Isaac-UAV-Position-RL-v0``): target position -> velocity, 50 Hz.

Frozen velocity, attitude and rate layers below; the command is a random target that runs on a circle or a figure 8 (or stays) and its
velocity, so the layer learns to follow trajectories. Everything else is
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
    """Position error, velocity, velocity of the target, last 2 outputs (15)."""

    position_error = ObsTerm(func=mdp.position_error)
    velocity = ObsTerm(func=isaac_mdp.root_lin_vel_w)
    target_velocity = ObsTerm(func=mdp.target_velocity)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class PositionRewardsCfg(RewardsCfg):
    # One term per axis (x, y, z): the weights sum to 1 for the coarse and for the fine terms, as before.
    position_x_coarse = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 0},
    )
    position_x_fine = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 0},
    )
    position_y_coarse = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 1},
    )
    position_y_fine = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 1},
    )
    position_z_coarse = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 1.0, "axis": 2},
    )
    position_z_fine = RewTerm(
        func=mdp.position_axis_error_exp,
        weight=1.0 / 3.0,
        params={"std": 0.25, "axis": 2},
    )
    # Penalties 5, 5 and 3 times heavier than in the first version (-0.1, -1, -0.3). The settling term was lowered to
    # -0.3 earlier because it fights the fast approach of a P controller; watch the step response when it is this heavy.
    position_error = RewTerm(
        func=mdp.position_error_l2,
        weight=-0.5,
    )
    position_overshoot = RewTerm(                     # step response: overshoot, and every swing back and forth
        func=mdp.position_overshoot,
        weight=-5.0,
    )
    position_settling = RewTerm(                      # step response: slow arrival, still hover
        func=mdp.position_settling,
        weight=-1.0,
        params={"std": 0.15},
    )


@configclass
class PositionEnvCfg(LayerEnvCfg):
    LAYER = "position"
    EPISODE_S = 8.0
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(
            resampling_time_range=(4.0, 6.0),
            path_types=("circle", "figure8"),         # the target runs on a path, or stays (p_static)
            radius_range=(0.3, 0.8),                  # [m]
            speed_range=(0.3, 0.8),                   # [m/s]
            p_static=0.2,
        ),
    )
    observations: ObservationsCfg = ObservationsCfg(policy=PositionPolicyCfg())
    rewards: PositionRewardsCfg = PositionRewardsCfg()
