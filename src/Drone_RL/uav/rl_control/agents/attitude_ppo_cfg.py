"""PPO settings of the attitude layer task (``Isaac-UAV-Attitude-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name, make_actor


@configclass
class AttitudePPORunnerCfg(LayerPPORunnerCfg):
    num_steps_per_env = 48
    max_iterations = 1000
    experiment_name = experiment_name("attitude")
    actor = make_actor(init_std=0.5, std_range=(0.05, 0.5))     # fast layer: keep exploring, like the rate layer
