"""PPO settings of the rate layer task (``Isaac-UAV-Rate-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name, make_actor


@configclass
class RatePPORunnerCfg(LayerPPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    experiment_name = experiment_name("rate")
    actor = make_actor(init_std=0.1, std_range=(0.02, 0.3))
