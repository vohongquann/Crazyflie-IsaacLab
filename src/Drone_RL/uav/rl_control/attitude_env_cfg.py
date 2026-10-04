"""Attitude layer task (``Isaac-UAV-Attitude-RL-v0``): acceleration and yaw -> body rates and thrust, 250 Hz.

Rate below: the frozen rate network (``AttitudeEnvCfg``) or the PID rate controller (``AttitudePIDRateEnvCfg``,
``Isaac-UAV-Attitude-PIDRate-RL-v0``). The command comes from the PID position and velocity layers. Everything else is
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
class AttitudePolicyCfg(PolicyCfg):
    """Wanted force in the body frame, yaw error, gravity direction in the body frame, body rates, last 2 outputs (18)."""

    wanted_force_body = ObsTerm(func=mdp.wanted_force_body)
    yaw_error = ObsTerm(func=mdp.yaw_error)
    gravity_in_body = ObsTerm(func=isaac_mdp.projected_gravity)
    body_rates = ObsTerm(func=isaac_mdp.base_ang_vel)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class AttitudeRewardsCfg(RewardsCfg):
    tilt_error = RewTerm(
        func=mdp.tilt_error_l2,
        weight=-1.0,
    )
    yaw = RewTerm(
        func=mdp.yaw_error_exp,
        weight=1.0,
        params={"std": 0.4},
    )
    acceleration = RewTerm(
        func=mdp.acceleration_error_exp,
        weight=1.0,
        params={"std": 2.0},
    )
    spin = RewTerm(
        func=mdp.body_rates_l2,
        weight=-0.01,
    )


@configclass
class AttitudeEnvCfg(LayerEnvCfg):
    LAYER = "attitude"
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(offset=(2.0, 2.0, 1.5, 0.0)),     # [m/s^2]
    )
    observations: ObservationsCfg = ObservationsCfg(policy=AttitudePolicyCfg())
    rewards: AttitudeRewardsCfg = AttitudeRewardsCfg()


@configclass
class AttitudePIDRateEnvCfg(AttitudeEnvCfg):
    """Same task, with the PID rate controller (``pid_control/rate.py``) under the policy, not the rate network."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.cascade.pid_rate = True
