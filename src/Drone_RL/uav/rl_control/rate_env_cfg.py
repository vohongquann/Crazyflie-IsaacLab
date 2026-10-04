"""Rate layer task (``Isaac-UAV-Rate-RL-v0``): body rates and thrust -> four motor commands, 500 Hz.

No frozen layer below; the command comes from the PID position, velocity and attitude layers. Everything else is
``cascade_env_cfg.py``.
"""
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.mdp.commands import LayerCommandCfg
from Drone_RL.uav.mdp.flight import WEIGHT_N
from Drone_RL.uav.mdp.layers import AttitudeLayer
from Drone_RL.uav.rl_control.cascade_env_cfg import (
    CommandsCfg,
    EventCfg,
    LayerEnvCfg,
    ObservationsCfg,
    PolicyCfg,
    RewardsCfg,
)


@configclass
class RatePolicyCfg(PolicyCfg):
    """Rate error, body rates, wanted thrust / weight, last 4 outputs (23)."""

    rate_error = ObsTerm(func=mdp.rate_error)
    body_rates = ObsTerm(func=isaac_mdp.base_ang_vel)
    thrust_ratio = ObsTerm(func=mdp.thrust_ratio)
    action_history = ObsTerm(func=mdp.action_history)


@configclass
class RateEventCfg(EventCfg):
    """The drone starts level, at rest and facing about the target yaw."""

    reset_drone = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.3, 0.3),
            },
            "velocity_range": {},
        },
    )


@configclass
class RateRewardsCfg(RewardsCfg):
    rate_error = RewTerm(
        func=mdp.rate_error_l2,
        weight=-0.02,
    )
    thrust = RewTerm(
        func=mdp.thrust_error_exp,
        weight=1.0,
        params={"std": 0.1},
    )
    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.2,
    )
    terminated = RewTerm(
        func=isaac_mdp.is_terminated,
        weight=-1000.0,
    )


@configclass
class RateEnvCfg(LayerEnvCfg):
    LAYER = "rate"
    EPISODE_S = 5.0
    # Gentle commands (``level_only``): the PID above only holds the drone, and random rate offsets make the command.
    # The offsets go up to what the attitude layer can send, (6, 6, 3) rad/s and half the weight. The PID attitude
    # (kp 8.6, 4) pulls back, so a held 6 rad/s offset settles at about 0.7 rad of tilt.
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(
            level_only=True,
            yaw_range=0.3,
            offset_time=(0.2, 0.8),
            offset=(*AttitudeLayer.RATE_SCALE, AttitudeLayer.THRUST_SCALE * WEIGHT_N),
        ),
    )
    events: RateEventCfg = RateEventCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=RatePolicyCfg())
    rewards: RateRewardsCfg = RateRewardsCfg()
