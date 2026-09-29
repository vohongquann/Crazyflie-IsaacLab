"""PPO settings of the attitude layer task (``Isaac-UAV-Attitude-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg
from Drone_RL.uav.rl_control.freeze import experiment_name


@configclass
class AttitudePPORunnerCfg(LayerPPORunnerCfg):
    max_iterations = 1000
    experiment_name = experiment_name("attitude")
