"""PPO settings of the attitude layer task (``Isaac-UAV-Attitude-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name


@configclass
class AttitudePPORunnerCfg(LayerPPORunnerCfg):
    max_iterations = 1000
    experiment_name = experiment_name("attitude")


@configclass
class AttitudePIDRatePPORunnerCfg(AttitudePPORunnerCfg):
    """``Isaac-UAV-Attitude-PIDRate-RL-v0``: own log folder, so these runs are never taken for the attitude layer."""

    experiment_name = experiment_name("attitude") + "_pidrate"
