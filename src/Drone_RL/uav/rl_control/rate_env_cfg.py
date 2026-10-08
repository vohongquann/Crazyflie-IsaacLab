"""Rate layer task: body rates and thrust -> four motor commands, 500 Hz (base of ``Isaac-UAV-Rate-Gains-v0``).

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
    """Rate error, body rates, wanted thrust / weight, last outputs (the gains)."""

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
    # The l2 term alone (-0.02) let the policy survive by hovering and ignoring the command (error 6 rad/s); the two
    # exp terms pay for following it, coarse far from the command and fine near it.
    rate_coarse = RewTerm(
        func=mdp.rate_error_exp,
        weight=1.0,
        params={"std": 4.0},
    )
    rate_fine = RewTerm(
        func=mdp.rate_error_exp,
        weight=1.0,
        params={"std": 1.0},
    )
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
    # Offsets of (4, 4, 2) rad/s, held 0.1 to 0.5 s; the PID attitude (kp 8.6, 4) pulls back. With what the attitude
    # layer can send, (6, 6, 3) rad/s and half the weight held up to 0.8 s, even the PID rate controller crashed in 40 %
    # of the episodes (too low): the thrust offset of -W/2 alone drops the drone 1.6 m.
    commands: CommandsCfg = CommandsCfg(
        layer=LayerCommandCfg(
            level_only=True,
            yaw_range=0.3,
            offset_time=(0.1, 0.5),
            offset=(4.0, 4.0, 2.0, 0.15 * WEIGHT_N),
        ),
    )
    events: RateEventCfg = RateEventCfg()
    observations: ObservationsCfg = ObservationsCfg(policy=RatePolicyCfg())
    rewards: RateRewardsCfg = RateRewardsCfg()

    def play_mode(self):
        """``play``: gentle commands the eye can follow. The training offsets (up to 4 rad/s, a new one every 0.1 to
        0.5 s) make the drone jerk back and forth even when it follows them; ``play --train_env_cfg`` shows those."""
        super().play_mode()
        self.commands.layer.offset = (1.5, 1.5, 1.0, 0.0)
        self.commands.layer.offset_time = (1.0, 2.0)
