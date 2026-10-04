"""Shared part of the four RL cascade tasks (``rate_env_cfg.py`` .. ``position_env_cfg.py``, trained in that order).

    scene:        ground, light, Crazyflie starting around 1.5 m
    command:      mdp.commands.LayerCommand (PID layers above + random target and offsets)
    action:       mdp.actions.CascadeAction (this layer, then the frozen layers below, then the motors)
    observation:  one term per entry of ``Layer.observation`` (mdp/layers.py), from mdp.observations or Isaac Lab
    reward:       mdp.rewards: error to the command (set per layer) and small smoothness and termination terms
    termination:  time out, or out of the flight envelope (below 0.2 m, above 4 m, 3 m away, upside down)

Physics runs at 1000 Hz; the policy at the rate of its layer (rate 500 Hz, attitude 250 Hz,
velocity 100 Hz, position 50 Hz).
"""
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.cascade_action import CascadeActionCfg
from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ
from Drone_RL.uav.mdp.commands import LayerCommandCfg
from Drone_RL.uav.mdp.layers import LAYERS

START_HEIGHT_M = 1.5
FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
"""``<layer>.pt`` of the four layers: the ``exported/policy.pt`` that ``play`` writes in the run folder, copied here."""


@configclass
class FlightSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0), color=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, visible_in_primary_ray=False),     # lights, but is no sky
    )
    robot = U.UAV_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=U.UAV_CFG.init_state.replace(pos=(0.0, 0.0, START_HEIGHT_M)),
    )


@configclass
class ActionsCfg:
    # layer: set by LayerEnvCfg
    cascade = CascadeActionCfg(
        asset_name="robot",
        frozen_dir=str(FROZEN_DIR),
    )


@configclass
class CommandsCfg:
    layer = LayerCommandCfg()      # layer: set by LayerEnvCfg


@configclass
class PolicyCfg(ObsGroup):
    """Observation of the policy, one vector. Each layer lists its terms in its env cfg."""

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_drone = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.3, 0.3),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.3, 0.3),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """What every layer pays; each layer adds the error of what it tracks (``mdp.<quantity>_error_l2`` or ``_exp``)."""

    output_change = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.05,
    )
    terminated = RewTerm(
        func=isaac_mdp.is_terminated,
        weight=-500.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=isaac_mdp.time_out,
        time_out=True,
    )
    too_low = DoneTerm(
        func=isaac_mdp.root_height_below_minimum,
        params={"minimum_height": 0.2},
    )
    tilted = DoneTerm(
        func=isaac_mdp.bad_orientation,
        params={"limit_angle": 1.57},
    )
    left_volume = DoneTerm(func=mdp.left_flight_volume)


@configclass
class LayerEnvCfg(ManagerBasedRLEnvCfg):
    """Shared part. A layer sets ``LAYER`` and ``EPISODE_S``, and overrides ``commands``, ``events`` and ``rewards``."""

    LAYER = "rate"
    EPISODE_S = 6.0

    scene: FlightSceneCfg = FlightSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        layer = LAYERS[self.LAYER]
        self.actions.cascade.layer = layer.name
        self.commands.layer.layer = layer.name
        self.sim.dt = 1.0 / FW_TICK_HZ                       # 1000 Hz physics
        self.decimation = int(round(FW_TICK_HZ / layer.hz))  # policy at the rate of its layer
        # Render (window, video) at most 50 times per simulated second: a render costs far more than an env step of the
        # fast layers (rate 2 ms), and one render per step made the drone run in slow motion in the window.
        self.sim.render_interval = max(self.decimation, int(FW_TICK_HZ // 120))
        self.episode_length_s = self.EPISODE_S
        # Video (--video) and the window follow the drone of environment 0.
        # self.viewer = ViewerCfg(
        #     eye=(0.6, 0.6, 0.3),
        #     lookat=(0.0, 0.0, 0.0),
        #     origin_type="asset_root",
        #     asset_name="robot",
        #     env_index=0,
        # )