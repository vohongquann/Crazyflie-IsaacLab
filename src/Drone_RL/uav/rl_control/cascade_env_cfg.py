"""Shared part of the four RL cascade tasks (``rate_env_cfg.py`` .. ``position_env_cfg.py``, trained in that order).

    scene:        ground, light, Crazyflie starting around 1.5 m
    command:      mdp.commands.LayerCommand (PID layers above + random target and offsets)
    action:       mdp.actions.CascadeAction (this layer, then the frozen layers below, then the motors)
    observation:  mdp.observations.layer_observation (built by mdp.layers.<Layer>.observe)
    reward:       mdp.rewards: tracking of the command minus small smoothness terms (set per layer)
    termination:  time out, or out of the flight envelope (below 0.2 m, above 4 m, 3 m away, upside down)

Physics runs at 500 Hz; the policy at the rate of its layer (rate 100 Hz, the others 50 Hz).
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav import mdp
from Drone_RL.uav.mdp.actions.cascade_action import CascadeActionCfg
from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ
from Drone_RL.uav.mdp.commands import LayerCommandCfg
from Drone_RL.uav.mdp.layers import LAYERS
from Drone_RL.uav.rl_control.freeze import FROZEN_DIR

START_HEIGHT_M = 1.5


@configclass
class FlightSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = U.UAV_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=U.UAV_CFG.init_state.replace(pos=(0.0, 0.0, START_HEIGHT_M)),
    )


@configclass
class ActionsCfg:
    cascade = CascadeActionCfg(asset_name="robot", layer="rate", frozen_dir=str(FROZEN_DIR))


@configclass
class CommandsCfg:
    layer = LayerCommandCfg(layer="rate", debug_vis=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        layer = ObsTerm(func=mdp.layer_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_drone = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                           "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.3, 0.3),
                               "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    left_envelope = DoneTerm(func=mdp.left_flight_envelope)


@configclass
class LayerEnvCfg(ManagerBasedRLEnvCfg):
    """Shared part; each layer below sets ``LAYER``, its rewards and its episode length."""

    LAYER = "rate"
    EPISODE_S = 6.0

    scene: FlightSceneCfg = FlightSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        layer = LAYERS[self.LAYER]
        self.actions.cascade.layer = layer.name
        self.commands.layer.layer = layer.name
        self.sim.dt = 1.0 / FW_TICK_HZ                          # 500 Hz physics
        self.decimation = int(round(FW_TICK_HZ / layer.hz))     # policy at the rate of its layer
        self.sim.render_interval = self.decimation
        self.episode_length_s = self.EPISODE_S
        # Video (--video) and the window follow the drone of environment 0.
        self.viewer = ViewerCfg(eye=(0.6, 0.6, 0.3), lookat=(0.0, 0.0, 0.0), origin_type="asset_root",
                                asset_name="robot", env_index=0)


def termination_penalty(weight: float) -> RewTerm:
    """Penalty on falling or leaving the envelope; ``weight`` is per second, the reward manager multiplies by dt."""
    return RewTerm(func=isaac_mdp.is_terminated, weight=weight)
