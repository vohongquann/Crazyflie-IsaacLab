"""Attitude layer task: roll, pitch, yaw and thrust -> body rates (thrust passes through), 250 Hz (base of
``Isaac-UAV-Attitude-Gains-v0``).

Rate below: the frozen rate gain network. The command comes from the PID position and velocity layers (the velocity layer writes
roll, pitch and thrust). Everything else is
``cascade_env_cfg.py``.
"""
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.commands import LayerCommandCfg
from Drone_RL.uav.rl_control.cascade_env_cfg import (
    CommandsCfg,
    LayerEnvCfg,
    ObservationsCfg,
    PolicyCfg,
    RewardsCfg,
)


@configclass
class AttitudePolicyCfg(PolicyCfg):
    """Roll, pitch and yaw error, wanted thrust / weight, gravity direction in the body frame, body rates, last
    outputs (the gains)."""

    attitude_error = ObsTerm(func=mdp.attitude_error)
    thrust_ratio = ObsTerm(func=mdp.thrust_ratio)
    gravity_in_body = ObsTerm(func=isaac_mdp.projected_gravity)
    body_rates = ObsTerm(func=isaac_mdp.base_ang_vel)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class AttitudeRewardsCfg(RewardsCfg):
    # At -0.05 the policy changed its output by 0.6 (summed squares) every 4 ms: the wanted rates chattered by about
    # +-0.4 of their range and the motors changed 18 times more per step than under the PID attitude layer.
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.5,
    )
    tilt_error = RewTerm(
        func=mdp.tilt_error_l2,
        weight=-1.0,
    )
    yaw = RewTerm(
        func=mdp.yaw_error_exp,
        weight=1.0,
        params={"std": 0.4},
    )
    # With the fine term alone the yaw error stayed at 0.6 rad (PID: 0.43): far from the target yaw it pays nothing.
    yaw_coarse = RewTerm(
        func=mdp.yaw_error_exp,
        weight=0.5,
        params={"std": 1.5},
    )
    roll = RewTerm(
        func=mdp.roll_error_exp,
        weight=0.5,
        params={"std": 0.15},
    )
    pitch = RewTerm(
        func=mdp.pitch_error_exp,
        weight=0.5,
        params={"std": 0.15},
    )
    spin = RewTerm(
        func=mdp.body_rates_l2,
        weight=-0.01,
    )


@configclass
class AttitudeEnvCfg(LayerEnvCfg):
    LAYER = "attitude"
    commands: CommandsCfg = CommandsCfg(
        # roll, pitch [rad] (a tilt of 2 m/s^2), yaw, thrust [N] (1.5 m/s^2 up or down)
        layer=LayerCommandCfg(offset=(0.25, 0.25, 0.0, 1.5 * U.DRONE_MASS_TOTAL_KG)),
    )
    observations: ObservationsCfg = ObservationsCfg(policy=AttitudePolicyCfg())
    rewards: AttitudeRewardsCfg = AttitudeRewardsCfg()
