"""Gain tasks (``Isaac-UAV-<Layer>-Gains-v0``): every layer is a network that writes the 9 PID gains of its layer
(``mdp/gains.py``, ``mdp/actions/gain_action.py``).

Each task is the task of the same layer in ``<layer>_env_cfg.py`` (same command, observation terms, rewards, resets,
terminations) with another action: ``GainCascadeAction`` in place of ``CascadeAction``. Trained bottom-up like the RL
cascade (rate, attitude, velocity, position), each over the frozen gain networks of the layers below
(``frozen/gains/<layer>.pt``). The action history in the observation is that of the 9 gains (2 steps, 18 values).
"""
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from Drone_RL.uav.mdp.actions.gain_action import GainCascadeActionCfg
from Drone_RL.uav.rl_control.cascade_env_cfg import FROZEN_GAINS_DIR
from Drone_RL.uav.rl_control.attitude_env_cfg import AttitudeEnvCfg, AttitudeRewardsCfg
from Drone_RL.uav.rl_control.position_env_cfg import PositionEnvCfg
from Drone_RL.uav.rl_control.rate_env_cfg import RateEnvCfg, RateRewardsCfg
from Drone_RL.uav.rl_control.velocity_env_cfg import VelocityEnvCfg

GAIN_CHANGE_WEIGHT = -0.05
"""Penalty of the change of the gains between two steps. The gains are slow (they act through a PID), so this is the
light weight of the base task, not the heavy ones the motor and rate outputs needed."""


@configclass
class GainsActionsCfg:
    cascade = GainCascadeActionCfg(asset_name="robot", frozen_dir=str(FROZEN_GAINS_DIR))      # layer: set by LayerEnvCfg


@configclass
class RateGainsRewardsCfg(RateRewardsCfg):
    output_change = RewTerm(func=isaac_mdp.action_rate_l2, weight=GAIN_CHANGE_WEIGHT)


@configclass
class AttitudeGainsRewardsCfg(AttitudeRewardsCfg):
    output_change = RewTerm(func=isaac_mdp.action_rate_l2, weight=GAIN_CHANGE_WEIGHT)


@configclass
class RateGainsEnvCfg(RateEnvCfg):
    """Rate PID gains. Observation 25: rate error, body rates, thrust / weight, last 2 gains."""

    actions: GainsActionsCfg = GainsActionsCfg()
    rewards: RateGainsRewardsCfg = RateGainsRewardsCfg()


@configclass
class AttitudeGainsEnvCfg(AttitudeEnvCfg):
    """Attitude PID gains over the frozen rate gain network. Observation 28."""

    actions: GainsActionsCfg = GainsActionsCfg()
    rewards: AttitudeGainsRewardsCfg = AttitudeGainsRewardsCfg()


@configclass
class VelocityGainsEnvCfg(VelocityEnvCfg):
    """Velocity PID gains over the frozen attitude and rate gain networks. Observation 27."""

    actions: GainsActionsCfg = GainsActionsCfg()


@configclass
class PositionGainsEnvCfg(PositionEnvCfg):
    """Position PID gains over the frozen velocity, attitude and rate gain networks. Observation 27."""

    actions: GainsActionsCfg = GainsActionsCfg()
