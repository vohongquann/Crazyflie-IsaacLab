"""Rate layer task (``Isaac-UAV-Rate-RL-v0``): body rates and thrust -> four motor commands, 100 Hz.

No frozen layer below; the command comes from the PID position, velocity and attitude layers.
Shared scene, command, action, observation, events and terminations: ``cascade_env_cfg.py``. Rewards: ``mdp/rewards.py``.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav.mdp.layers import WEIGHT_N, AttitudeLayer
from Drone_RL.uav.rl_control.cascade_env_cfg import LayerEnvCfg, termination_penalty


@configclass
class RateRewardsCfg:
    rate_coarse = RewTerm(func=mdp.body_rate_tracking, weight=1.0, params={"std": 4.0})
    rate_fine = RewTerm(func=mdp.body_rate_tracking, weight=1.0, params={"std": 1.0})
    rate_error = RewTerm(func=mdp.body_rate_error_l2, weight=-0.02)
    thrust = RewTerm(func=mdp.thrust_tracking, weight=1.0, params={"std": 0.1})
    output_change = RewTerm(func=mdp.output_change, weight=-0.2)
    terminated = termination_penalty(-1000.0)


@configclass
class RateEnvCfg(LayerEnvCfg):
    LAYER = "rate"
    EPISODE_S = 5.0
    rewards: RateRewardsCfg = RateRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Gentle commands (commands.LayerCommandCfg.level_only): the PID above only holds the drone, random rate offsets,
        # a start that is level, at rest and facing about the target yaw.
        self.commands.layer.level_only = True
        self.commands.layer.yaw_range = 0.3
        self.commands.layer.offset_time = (0.2, 0.8)
        pose = self.events.reset_drone.params["pose_range"]
        pose.update({"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
                     "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.3, 0.3)})
        self.events.reset_drone.params["velocity_range"] = {}
        # Offsets on the PID command up to what the attitude layer can send: (6, 6, 3) rad/s and half the weight. The PID
        # attitude (kp 17.3, 4) pulls back, so a held 6 rad/s offset settles at about 0.35 rad of tilt.
        self.commands.layer.offset = (*AttitudeLayer.RATE_SCALE, AttitudeLayer.THRUST_SCALE * WEIGHT_N)
