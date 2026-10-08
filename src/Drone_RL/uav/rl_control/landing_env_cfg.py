"""Landing task (``Isaac-UAV-Landing-ArUco-v0``): fly the Crazyflie from a random point above the pad down onto
the ArUco marker.

    policy:   position, orientation, velocities and action history (Eschmann et al. 2024),
              plus the ArUco detection (u, v, size, found)
    action:   four motor commands (MotorAction), centred on the hover throttle
    reward:   stay above the pad, go down while aligned, touch down slowly, land on the marker (bonus)
    episode:  10 s, ends early on landing (success) or crash

Details, frames and numbers: guide/06_aruco_landing.md.
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from Drone_RL.uav import mdp
from Drone_RL.uav import uav_cfg as U
from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ, LANDING_HZ
from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
from Drone_RL.uav.rl_control.marker_plate import ArucoPlateCfg

MARKER_SIZE_M = 0.8          # side of the pad including the white margin; the marker itself is 2/3 of it
CAMERA_PIXELS = 192
ACTION_HISTORY = 4         # motor commands of the last 4 policy steps (80 ms) in the observation
START_HEIGHT_M = 1.4         # upper end of the random starting height


@configclass
class LandingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0),
    )
    marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Marker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002)),
        spawn=ArucoPlateCfg(size=MARKER_SIZE_M),
    )
    robot = U.UAV_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Looks straight down, x of the image = -y of the drone, up in the image = x of the drone (ROS camera convention).
    # Field of view 90 degrees: focal length 10 mm on a 20 mm aperture.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        update_period=0.0,
        width=CAMERA_PIXELS,
        height=CAMERA_PIXELS,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0,
            horizontal_aperture=20.0,
            clipping_range=(0.02, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, -0.01),
            rot=(0.7071068, -0.7071068, 0.0, 0.0),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    motor = MotorActionCfg(
        asset_name="robot",
        offset=U.DRONE_HOVER_THROTTLE,
        scale=0.25,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # State list of Eschmann et al. 2024 (mdp/observations.py) ...
        position = ObsTerm(func=mdp.position_relative_to_target)
        orientation = ObsTerm(
            func=isaac_mdp.root_quat_w,
            params={"make_quat_unique": True},
        )
        linear_velocity = ObsTerm(func=isaac_mdp.root_lin_vel_w)
        angular_velocity = ObsTerm(func=isaac_mdp.base_ang_vel)
        action_history = ObsTerm(
            func=isaac_mdp.last_action,
            params={"action_name": "motor"},
            history_length=ACTION_HISTORY,
        )
        # ... plus what the downward camera says about the marker.
        aruco = ObsTerm(
            func=mdp.ArucoObservation,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

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
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.7, START_HEIGHT_M - 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
        },
    )


@configclass
class RewardsCfg:
    alignment = RewTerm(
        func=mdp.alignment,
        weight=1.0,
    )
    descent = RewTerm(
        func=mdp.descent_over_pad,
        weight=6.0,
        params={"start_height": START_HEIGHT_M},
    )
    soft_touchdown = RewTerm(
        func=mdp.touchdown_speed_penalty,
        weight=-5.0,
    )
    upright = RewTerm(
        func=isaac_mdp.flat_orientation_l2,
        weight=-1.0,
    )
    spin = RewTerm(
        func=isaac_mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    action_rate = RewTerm(
        func=isaac_mdp.action_rate_l2,
        weight=-0.01,
    )
    landed = RewTerm(
        func=isaac_mdp.is_terminated_term,
        weight=2500.0,
        params={"term_keys": "landed"},
    )
    # As large as the landing bonus: with -1000 a fast dive that lands half the time still paid, so the policy
    # gambled (73 % landed at best, landing in 2 s).
    crashed = RewTerm(
        func=isaac_mdp.is_terminated_term,
        weight=-2500.0,
        params={"term_keys": "crashed"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=isaac_mdp.time_out,
        time_out=True,
    )
    landed = DoneTerm(func=mdp.landed)
    crashed = DoneTerm(func=mdp.crashed)


@configclass
class LandingEnvCfg(ManagerBasedRLEnvCfg):
    scene: LandingSceneCfg = LandingSceneCfg(
        num_envs=16,
        env_spacing=4.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / FW_TICK_HZ                          # 1000 Hz physics
        self.decimation = int(round(FW_TICK_HZ / LANDING_HZ))   # 25 Hz policy and camera
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
