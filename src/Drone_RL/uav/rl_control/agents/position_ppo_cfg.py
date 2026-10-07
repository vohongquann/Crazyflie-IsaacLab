"""PPO settings of the position layer task (``Isaac-UAV-Position-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name, make_actor


@configclass
class PositionPPORunnerCfg(LayerPPORunnerCfg):
    max_iterations = 1000
    experiment_name = experiment_name("position")
    actor = make_actor(init_std=0.2)
