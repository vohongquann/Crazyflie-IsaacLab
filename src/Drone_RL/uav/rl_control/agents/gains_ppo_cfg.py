"""PPO settings of the gain tasks (``Isaac-UAV-<Layer>-Gains-v0``): those of the task of the same layer, with the action
noise of a gain schedule. The gains act through a PID, so the noise of a gain (the std is in the exponent of 3^a) stays
small: 0.3 at most. Each task has its own log folder ``uav_<layer>_gains``."""
from isaaclab.utils import configclass

from Drone_RL.uav.rl_control.agents.attitude_ppo_cfg import AttitudePPORunnerCfg
from Drone_RL.uav.rl_control.agents.cascade_ppo_cfg import experiment_name, make_actor
from Drone_RL.uav.rl_control.agents.position_ppo_cfg import PositionPPORunnerCfg
from Drone_RL.uav.rl_control.agents.rate_ppo_cfg import RatePPORunnerCfg
from Drone_RL.uav.rl_control.agents.velocity_ppo_cfg import VelocityPPORunnerCfg


def gains_actor():
    return make_actor(init_std=0.2, std_range=(0.02, 0.3))


@configclass
class RateGainsPPORunnerCfg(RatePPORunnerCfg):
    experiment_name = experiment_name("rate") + "_gains"
    actor = gains_actor()


@configclass
class AttitudeGainsPPORunnerCfg(AttitudePPORunnerCfg):
    experiment_name = experiment_name("attitude") + "_gains"
    actor = gains_actor()


@configclass
class VelocityGainsPPORunnerCfg(VelocityPPORunnerCfg):
    experiment_name = experiment_name("velocity") + "_gains"
    actor = gains_actor()


@configclass
class PositionGainsPPORunnerCfg(PositionPPORunnerCfg):
    experiment_name = experiment_name("position") + "_gains"
    actor = gains_actor()
