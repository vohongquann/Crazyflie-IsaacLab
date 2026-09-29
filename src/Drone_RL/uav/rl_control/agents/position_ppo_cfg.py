"""PPO settings of the position layer task (``Isaac-UAV-Position-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, make_actor
from Drone_RL.uav.rl_control.freeze import experiment_name


@configclass
class PositionPPORunnerCfg(LayerPPORunnerCfg):
    experiment_name = experiment_name("position")
    actor = make_actor(init_std=0.2)
