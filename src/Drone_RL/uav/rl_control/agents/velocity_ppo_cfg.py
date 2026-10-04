"""PPO settings of the velocity layer task (``Isaac-UAV-Velocity-RL-v0``); the rest is in ``cascade_ppo_cfg.py``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import LayerPPORunnerCfg, experiment_name, make_actor


@configclass
class VelocityPPORunnerCfg(LayerPPORunnerCfg):
    experiment_name = experiment_name("velocity")
    actor = make_actor(init_std=0.2)
