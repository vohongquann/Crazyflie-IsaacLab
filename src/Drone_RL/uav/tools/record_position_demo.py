"""Record a demo clip of the position layer (gain network over the frozen gain networks), for the README.

One drone hovers, then follows a circle and a figure 8 (the targets move, so the network gets the target velocity too).
The camera is scripted: it orbits the drone, changing distance and height, and always looks at it. The scene is drawn
for the picture: a gently uneven terrain (difficulty level 1 of 10) under an HDR sky, a sun (or a moon with
``--sky night``) that casts shadows, ray tracing (RTX real-time path tracing; ``--path_tracing`` for the slow,
offline-quality one), DLAA and turning propellers. The path is drawn ahead of the drone, a few seconds before it
flies it (blue dots), and the flown trail in cyan; both stay hidden during the first ``HIDE_FIRST`` seconds.

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/tools/record_position_demo.py \\
        --policy src/Drone_RL/uav/rl_control/frozen/gains/position.pt

The three layers below (``frozen/gains/velocity.pt``, ``attitude.pt``, ``rate.pt``) must exist. The clip is written to
``videos/demo/position_demo.mp4`` (2560x1440, 50 fps, real time; ``--width 3840 --height 2160`` for 4K). Try
``--seconds 6 --width 960 --height 540`` first.

README: GitHub shows an mp4 dragged into the editor. As a GIF or a smaller file:

    ffmpeg -i videos/demo/position_demo.mp4 -vf "fps=25,scale=960:-1:flags=lanczos" -loop 0 demo.gif
    ffmpeg -i videos/demo/position_demo.mp4 -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart demo_small.mp4
"""
import argparse
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parents[1]

# ── Command line ──────────────────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default=str(HERE / "rl_control" / "frozen" / "gains" / "position.pt"),
                    help="exported policy.pt of Isaac-UAV-Position-Gains-v0")
parser.add_argument("--out", default="videos/demo", help="folder of the clip")
parser.add_argument("--name", default="position_demo", help="file name: <name>.mp4")
parser.add_argument("--seconds", type=float, default=None, help="length of the clip (default: the whole program)")
parser.add_argument("--width", type=int, default=2560)
parser.add_argument("--height", type=int, default=1440)
parser.add_argument("--fps", type=int, default=50, help="50 is real time (the position layer runs at 50 Hz)")
parser.add_argument("--light", type=float, default=1.0, help="scales the sun and the sky light: 0.5 darker, 2 brighter")
parser.add_argument("--sky", default="day", choices=["night", "day", "color"],
                    help="night / day: sky texture of the Isaac Sim assets (downloaded the first time); "
                         "color: plain dark sky, no download")
parser.add_argument("--load_wait", type=float, default=15.0,
                    help="seconds to render before the clip starts, so the sky texture is loaded (it streams in)")
parser.add_argument("--path_tracing", action="store_true",
                    help="full path tracing with the denoiser: best picture, several times slower")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.video = True            # the viewport must render, also headless
app = AppLauncher(args).app

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import SPHERE_MARKER_CFG  # noqa: E402
from isaaclab.terrains import HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg  # noqa: E402
from isaaclab.terrains import MeshRandomGridTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

from Drone_RL.uav.rl_control.gains_env_cfg import PositionGainsEnvCfg  # noqa: E402

torch.set_grad_enabled(False)

# ── Look ──────────────────────────────────────────────────────────────────────────────────────────────────────
BACKGROUND = (0.015, 0.02, 0.04)             # sky colour with --sky color
GROUND = (0.10, 0.10, 0.11)                  # colour of the ground (dark grey)
TERRAIN_LEVEL = 1                            # difficulty level of the terrain, of 10
SKIES = {   # sky: (texture, sky light intensity, sun or moon intensity, its colour)
    "night": (f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Night/moonlit_golf_4k.hdr", 60.0, 200.0, (0.62, 0.72, 1.0)),
    "day": (f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            300.0, 3000.0, (1.0, 0.95, 0.88)),
}
PATH_COLOR, TRAIL_COLOR = (0.25, 0.55, 1.0), (0.15, 0.95, 1.0)
PATH_RADIUS, TRAIL_RADIUS = 0.009 / 2.25, 0.011 / 2.25      # [m] radius of a dot
PATH_SPACING = 0.12                          # [s] one dot of the path per this much program time
TRAIL_POINTS, TRAIL_EVERY = 90, 2            # 90 dots, one every 2 steps: the last 3.6 s
HIDE_FIRST = 2.0                             # [s] no path and no trail at the start of the clip
LOOKAHEAD = 1.5                              # [s] a dot of the path appears this long before the drone flies it

# ── The program (relative to the environment origin; the drone starts at the hover point) ─────────────────────────
STEP_DT = 0.02                               # [s] one step of the position layer
HEIGHT = 1.1                                 # [m] flight height, about 1 m above the terrain
HOVER = np.array([0.0, 0.0, HEIGHT])
HOVER_TIME = 3.0                             # [s] hover before the circle
CIRCLE_RADIUS, CIRCLE_SPEED = 0.8, 0.7       # [m], [m/s]
CIRCLE_PERIOD = 2.0 * math.pi * CIRCLE_RADIUS / CIRCLE_SPEED
EIGHT_START = HOVER_TIME + CIRCLE_PERIOD
EIGHT_SIZE, EIGHT_RATE = 1.0, 0.8            # [m], phase rate [rad/s]
EIGHT_PERIOD = 2.0 * math.pi / EIGHT_RATE
PROGRAM_END = EIGHT_START + EIGHT_PERIOD + 1.5


def target(t: float):
    """Position and velocity of the target at time ``t`` of the program: hover, circle, figure 8, hover."""
    if t < HOVER_TIME:
        return HOVER, np.zeros(3)
    if t < EIGHT_START:                                          # circle, starts at (0, 0)
        phase = CIRCLE_SPEED / CIRCLE_RADIUS * (t - HOVER_TIME)
        position = (-CIRCLE_RADIUS + CIRCLE_RADIUS * math.cos(phase), CIRCLE_RADIUS * math.sin(phase), HEIGHT)
        velocity = (-CIRCLE_SPEED * math.sin(phase), CIRCLE_SPEED * math.cos(phase), 0.0)
        return np.array(position), np.array(velocity)
    phase = EIGHT_RATE * min(t - EIGHT_START, EIGHT_PERIOD)      # figure 8 (lemniscate of Gerono), starts at (0, 0)
    position = (EIGHT_SIZE * math.sin(phase), EIGHT_SIZE * 0.5 * math.sin(2.0 * phase), HEIGHT + 0.1 * math.sin(phase))
    velocity = (EIGHT_SIZE * EIGHT_RATE * math.cos(phase), EIGHT_SIZE * EIGHT_RATE * math.cos(2.0 * phase),
                0.1 * EIGHT_RATE * math.cos(phase))
    if t - EIGHT_START >= EIGHT_PERIOD:
        velocity = (0.0, 0.0, 0.0)
    return np.array(position), np.array(velocity)


# ── Camera: keyframes (share of the program, azimuth [deg], distance [m], height [m]) around the drone ────────
CAMERA_KEYS = [(0.0, -70.0, 3.0, 1.9), (0.19, -20.0, 2.2, 1.3), (0.38, 40.0, 1.9, 1.1), (0.5, 100.0, 2.2, 1.8),
               (0.75, 200.0, 2.0, 1.3), (1.0, 300.0, 3.0, 2.2)]


def smooth_keys(share: float, keys):
    """Interpolate the keyframes with a smoothstep between two of them (no jerk at the keys)."""
    if share <= keys[0][0]:
        return np.array(keys[0][1:], dtype=float)
    for (s0, *a), (s1, *b) in zip(keys[:-1], keys[1:]):
        if share < s1:
            u = (share - s0) / (s1 - s0)
            u = u * u * (3.0 - 2.0 * u)
            return (1.0 - u) * np.array(a) + u * np.array(b)
    return np.array(keys[-1][1:], dtype=float)


def camera_pose(t: float, drone: np.ndarray):
    """Camera position and the point it looks at: the scripted orbit, always looking at the drone."""
    azimuth, distance, height = smooth_keys(t / PROGRAM_END, CAMERA_KEYS)
    a = math.radians(azimuth)
    eye = np.array([distance * math.cos(a), distance * math.sin(a), height]) + 0.5 * np.array([drone[0], drone[1], 0.0])
    return eye, drone


# ── Scene ─────────────────────────────────────────────────────────────────────────────────────────────────────
def make_ground() -> TerrainImporterCfg:
    """A gently uneven terrain (boxes, noise, slopes) that fills the view: 15x15 tiles of 5 m (75 m) centred on the
    drone, then a flat border. At level 1 it is at most 0.1 m high, far below the drone (it flies
    at 1.1 m)."""
    level = TERRAIN_LEVEL / 10.0
    generator = TerrainGeneratorCfg(
        size=(5.0, 5.0), border_width=40.0, num_rows=15, num_cols=15, horizontal_scale=0.15, vertical_scale=0.005,
        slope_threshold=0.75, use_cache=False, seed=3, difficulty_range=(level, level),
        sub_terrains={
            "boxes": MeshRandomGridTerrainCfg(
                proportion=0.4, grid_width=0.4, grid_height_range=(0.05, 0.2), platform_width=0.0,
                convert_to_heightfield=True),
        })
    return TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", use_terrain_origins=False, terrain_generator=generator,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=GROUND, roughness=1.0, metallic=0.0))


def make_ground_matte() -> None:
    """No specular reflection on the ground: seen at a grazing angle it reflects the sky as a bright haze."""
    from pxr import Sdf, Usd, UsdShade

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    shaders = [UsdShade.Shader(prim) for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/ground"))
               if prim.IsA(UsdShade.Shader)]
    for shader in shaders:
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.0, 0.0))
    print(f"DEMO ground shaders made matte: {[str(shader.GetPath()) for shader in shaders]}")


def make_cfg(steps: int) -> PositionGainsEnvCfg:
    cfg = PositionGainsEnvCfg()
    cfg.scene.num_envs = 1
    cfg.episode_length_s = steps * STEP_DT + 10.0
    cfg.events.reset_drone.params["pose_range"] = {}
    cfg.events.reset_drone.params["velocity_range"] = {}
    for name in ("time_out", "too_low", "tilted", "left_volume"):
        setattr(cfg.terminations, name, None)
    cfg.commands.layer.debug_vis = False                 # this script draws its own path
    cfg.actions.cascade.spin_propellers = True           # turning propellers (one environment: cheap)
    cfg.scene.ground = make_ground()
    robot = cfg.scene.robot
    cfg.scene.robot = robot.replace(init_state=robot.init_state.replace(pos=tuple(HOVER)))     # starts at the flight height

    if args.sky != "color":                              # the sky is the dome texture, seen behind the terrain
        texture, sky_light, sun_light, sun_color = SKIES[args.sky]
        sky = sim_utils.DomeLightCfg(
            intensity=sky_light * args.light, texture_file=texture, visible_in_primary_ray=True)
    else:
        sun_light, sun_color = SKIES["night"][2:]
        sky = sim_utils.DomeLightCfg(intensity=40.0 * args.light, color=(0.5, 0.6, 1.0), visible_in_primary_ray=True)
    cfg.scene.light = AssetBaseCfg(prim_path="/World/light", spawn=sky)
    cfg.scene.sun = AssetBaseCfg(       # the sun or the moon: the only light that casts a shadow of the drone
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=sun_light * args.light, angle=0.6, color=sun_color),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3827, 0.0, 0.0, 0.9239)))         # 45 degrees about x

    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
        eye=(3.0, -3.0, 1.9), lookat=tuple(HOVER), window_width=args.width, window_height=args.height,
        background_color=BACKGROUND if args.sky == "color" else None)]      # a colour here would hide the sky texture
    # The recorder of Isaac Lab keeps every frame in memory until the end (1600 frames of 2560x1440 are 18 GB): it only
    # grabs frames here (step_offset keeps it from recording by itself), and each frame goes straight to ffmpeg.
    cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=args.out, step_offset=10**9)]
    return cfg


def set_render_quality() -> None:
    """Maximum picture quality: ray tracing, DLAA, shadows, ambient occlusion, global illumination."""
    try:
        import carb.settings

        quality = {
            "/rtx/post/aa/op": 4, "/rtx/post/histogram/enabled": True,       # DLAA; fixed exposure
            "/rtx/shadows/enabled": True, "/rtx/ambientOcclusion/enabled": True, "/rtx/indirectDiffuse/enabled": True,
            "/rtx/directLighting/sampledLighting/enabled": True, "/rtx/raytracing/subpixel/mode": 2,
            "/rtx/rtpt/maxBounces": 8, "/rtx/rtpt/adaptiveSampling/disocclusion/spp": 8}
        if args.path_tracing:
            quality.update({"/rtx/rendermode": "PathTracing", "/rtx/pathtracing/spp": 16,
                            "/rtx/pathtracing/totalSpp": 16, "/rtx/pathtracing/maxBounces": 8,
                            "/rtx/pathtracing/optixDenoiser/enabled": True})
        else:
            quality["/rtx/rendermode"] = "RealTimePathTracing"
        settings = carb.settings.get_settings()
        for key, value in quality.items():
            settings.set(key, value)
    except Exception as error:      # noqa: BLE001
        print("DEMO render settings skipped:", error)


class Dots:
    """A group of small glowing spheres (path or trail), hidden while it has no point."""

    def __init__(self, prim_path: str, radius: float, color: tuple, origin: np.ndarray, device, moving: bool = False):
        marker = SPHERE_MARKER_CFG.copy()
        marker.prim_path = prim_path
        marker.markers["sphere"].radius = radius
        marker.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, emissive_color=tuple(0.5 * c for c in color))
        self.markers = VisualizationMarkers(marker)
        self.markers.set_visibility(False)
        self.origin, self.device, self.moving = origin, device, moving
        self.count = 0

    def show(self, points: np.ndarray) -> None:
        """Draw these points. A fixed path is redrawn only when it grows; a moving trail every call."""
        if len(points) == self.count and not self.moving:
            return
        self.count = len(points)
        self.markers.set_visibility(self.count > 0)
        if self.count > 0:
            translations = torch.tensor(points + self.origin, dtype=torch.float32, device=self.device)
            self.markers.visualize(translations=translations)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ── Run ───────────────────────────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    seconds = args.seconds or PROGRAM_END
    steps, warm_steps = int(round(seconds / STEP_DT)), 10
    env = ManagerBasedRLEnv(make_cfg(steps))
    make_ground_matte()
    set_render_quality()
    device = env.device
    origin = env.scene.env_origins[0].cpu().numpy()

    trail = Dots("/Visuals/Demo/trail", TRAIL_RADIUS, TRAIL_COLOR, origin, device, moving=True)
    path = Dots("/Visuals/Demo/path", PATH_RADIUS, PATH_COLOR, origin, device)
    path_times = np.arange(HOVER_TIME, PROGRAM_END, PATH_SPACING)       # the time each dot of the path is flown
    path_points = np.array([target(t)[0] for t in path_times])

    policy = torch.jit.load(args.policy, map_location=device)
    env.reset()
    if args.sky != "color":     # without the wait the first frames have no sky texture
        deadline = time.time() + args.load_wait
        while time.time() < deadline:
            app.update()
    command_term = env.command_manager.get_term("layer")
    command_term._update_command = lambda: None         # the command is the program, not a random target

    Path(args.out).mkdir(parents=True, exist_ok=True)
    clip = Path(args.out) / f"{args.name}.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(clip), (args.width, args.height), fps=args.fps, codec="libx264", quality=None, macro_block_size=1,
        output_params=["-crf", "16", "-preset", "slow", "-movflags", "+faststart"])
    writer.send(None)

    trail_points: list = []
    drone_smooth, eye_smooth = HOVER.copy(), None
    yaw = 0.0
    for k in range(steps + warm_steps):
        t = max(0.0, (k - warm_steps) * STEP_DT)
        position, velocity = target(t)
        if np.linalg.norm(velocity[:2]) > 0.15:             # turn the nose along the flight
            wanted = math.atan2(velocity[1], velocity[0])
            yaw += float(np.clip(wrap_angle(wanted - yaw), -1.2 * STEP_DT, 1.2 * STEP_DT))
        command_term._command[:] = torch.tensor([*position, yaw, *velocity], dtype=torch.float32, device=device)
        obs = env.observation_manager.compute()["policy"]

        drone = isaac_mdp.root_pos_w(env)[0].cpu().numpy()
        drone_smooth += 0.08 * (drone - drone_smooth)
        eye, look = camera_pose(t, drone_smooth)
        eye_smooth = eye if eye_smooth is None else eye_smooth + 0.15 * (eye - eye_smooth)
        env.sim.set_camera_view(tuple(float(v) for v in eye_smooth), tuple(float(v) for v in look))

        if t >= HIDE_FIRST:
            path.show(path_points[path_times <= t + LOOKAHEAD])
            if k % TRAIL_EVERY == 0:
                trail_points = (trail_points + [drone])[-TRAIL_POINTS:]
                trail.show(np.array(trail_points))

        env.step(policy(obs).clamp(-1.0, 1.0))
        if k >= warm_steps:
            frame = env.video_recorders[0]._get_frame()
            if frame is not None:
                writer.send(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8))
    writer.close()
    env.close()
    print(f"DEMO written to {clip}")


main()
app.close()
