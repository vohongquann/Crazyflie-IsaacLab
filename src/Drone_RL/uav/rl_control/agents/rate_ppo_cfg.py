"""PPO settings of the rate layer task (``Isaac-UAV-Rate-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name, make_actor


@configclass
class RatePPORunnerCfg(LayerPPORunnerCfg):
    num_steps_per_env = 64
    max_iterations = 1500
    experiment_name = experiment_name("rate")
    # At 500 Hz a motor noise of 0.02 (the old lower bound) explores almost nothing: the std sat at the bound by
    # iteration 300 and the rate error stayed at 5 rad/s. The bound is now 0.05.
    actor = make_actor(init_std=0.3, std_range=(0.05, 0.5))
