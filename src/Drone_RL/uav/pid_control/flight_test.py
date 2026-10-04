"""Runner shared by the layer tests (rate.py, attitude.py, velocity.py, position.py): fly one layer in Isaac Sim.

Each layer file defines a ``LayerTest`` (setpoint steps, how to drive the layers, what to plot) and calls ``run`` from its
``__main__``. ``run`` starts Isaac Sim, flies the test on the Crazyflie USD with the same ``Propulsion`` as the RL action
(no drag, no thrust noise, true simulator state), prints the tracking error and draws actual vs target:

    --viz kit    Isaac Sim window (default; ``--viz none`` for none); the flight then runs at real-time speed and the
                 camera follows the drone
    --live       plot redrawn while flying (default; ``--no_live`` turns it off)
    --warehouse  small warehouse around the drone (window only; hundreds of MB downloaded from the NVIDIA server the
                 first time, then cached)
    --save PATH  PNG written at the end (default ``pid_<layer>.png`` in the current directory)

Isaac Lab is imported inside the functions, after the app has started, so the layer files stay importable in tests.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch

LOG_EVERY_S = 1.0       # one line of text per second of flight
PLOT_EVERY_S = 0.1      # live plot redraw: it takes tens of ms and would stall the real-time loop if done every step
RENDER_EVERY_S = 0.016  # window mode: one frame per 16 ms (62 Hz); rendering at every 2 ms physics step is too slow
CAMERA_OFFSET = (-1.2, -1.2, 0.6)   # [m] camera position relative to the point it looks at
CAMERA_SMOOTH = 0.1     # the look-at point covers this fraction of its distance to the drone at every frame
CRASH_Z = -0.05         # [m] below this the drone has hit the ground
MIN_SPAN = 1.0
"""Smallest range of a plot panel (in the unit of the panel). A panel whose value stays at zero would otherwise be
autoscaled to its float noise (1e-5 deg/s) and look like a violent shaking."""

WAREHOUSE_USD = "Environments/Simple_Warehouse/warehouse.usd"     # under ``ISAAC_NUCLEUS_DIR``, 15 x 25 m


@dataclass
class LayerTest:
    name: str                                  # "rate", "attitude", ...
    title: str
    labels: list                               # one plot panel per value, with the unit
    start_z: float                             # [m] height at which the drone starts
    duration: float                            # [s]
    setpoint: Callable                         # t [s] -> list of wanted values (in the units of ``labels``)
    control: Callable                          # (controller, wanted, state, dt) -> PWM (1, 4)
    measure: Callable                          # state -> list of measured values (same units as ``setpoint``)


# ----------------------------------------------------------------------------------------------------------------------
# Helpers for the layer files
# ----------------------------------------------------------------------------------------------------------------------

def euler_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternion (x, y, z, w) -> (N, 3) roll, pitch, yaw [rad], rotation order Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z, w = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack([roll, pitch, yaw], dim=-1)


def step_table(steps: list, t: float) -> list:
    """``steps`` = [(t_start, v1, v2, ...), ...] sorted by time: the values of the last step started at ``t``."""
    current = steps[0]
    for row in steps:
        if t >= row[0]:
            current = row
    return list(current[1:])


# ----------------------------------------------------------------------------------------------------------------------
# Plot and report
# ----------------------------------------------------------------------------------------------------------------------

def make_panels(test: LayerTest):
    """Empty figure with one panel per value. Returns ``(fig, axes, lines)``, ``lines[i] = (actual, setpoint)``."""
    import matplotlib.pyplot as plt

    count = len(test.labels)
    fig, axes = plt.subplots(count, 1, figsize=(10, 2.8 * count), sharex=True, squeeze=False)
    axes = axes[:, 0]
    lines = []
    for ax, label in zip(axes, test.labels):
        (actual,) = ax.plot([], [], "b-", lw=1.8, label="actual")
        (setpoint,) = ax.plot([], [], "r--", lw=1.4, label="setpoint")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        lines.append((actual, setpoint))
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(test.title, fontsize=13, fontweight="bold")
    return fig, axes, lines


def draw(axes, lines, times: list, actual: list, target: list) -> None:
    """Put the flight so far into the panels and rescale them."""
    for i, (ax, (line_actual, line_setpoint)) in enumerate(zip(axes, lines)):
        line_actual.set_data(times, [row[i] for row in actual])
        line_setpoint.set_data(times, [row[i] for row in target])
        ax.relim()
        ax.autoscale_view()
        low, high = ax.get_ylim()
        if high - low < MIN_SPAN:                                       # keep a flat panel flat
            centre = 0.5 * (low + high)
            ax.set_ylim(centre - 0.5 * MIN_SPAN, centre + 0.5 * MIN_SPAN, auto=True)   # auto: keep rescaling later


def save_plot(test: LayerTest, times: list, actual: list, target: list, path: str) -> None:
    """PNG with the whole flight: actual (blue) against the setpoint (red dashed)."""
    import matplotlib.pyplot as plt

    fig, axes, lines = make_panels(test)
    draw(axes, lines, times, actual, target)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(test: LayerTest, actual: list, target: list) -> None:
    """RMS and peak tracking error of every value."""
    error = torch.tensor(actual) - torch.tensor(target)
    rms = error.pow(2).mean(0).sqrt()
    peak = error.abs().max(0).values
    for label, r, p in zip(test.labels, rms.tolist(), peak.tolist()):
        print(f"  {label:<18} RMS error {r:8.3f}   max |error| {p:8.3f}")


def log_line(test: LayerTest, t: float, measured: list, wanted: list) -> None:
    columns = [f"{label.split(' [')[0]}={m:+8.2f} (want {w:+7.2f})"
               for label, m, w in zip(test.labels, measured, wanted)]
    print(f"t={t:5.1f}s  " + "  ".join(columns))


# ----------------------------------------------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------------------------------------------

def parse_args(test: LayerTest):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=test.title)
    parser.add_argument("--duration", type=float, default=test.duration, help="Flight time [s].")
    parser.add_argument("--live", action="store_true", default=True, help="Redraw the plot while flying (default).")
    parser.add_argument("--no_live", dest="live", action="store_false", help="No live plot (PNG only).")
    parser.add_argument("--warehouse", action="store_true", help="Put the drone in a small warehouse (needs network).")
    parser.add_argument("--save", type=str, default=f"pid_{test.name}.png", help="PNG written at the end.")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])     # Isaac Sim window by default; --viz none for no window
    args = parser.parse_args()
    if args.warehouse and not args.kit_args:
        # The first download of the warehouse blocks Kit for minutes; without this its hang detector asks
        # "Kit appears to be hanging, terminate?" and writes a crash report if the answer is yes.
        args.kit_args = "--/app/hangDetector/enabled=false"
    return args


def build_scene(test: LayerTest, dt: float, device: str, warehouse: bool):
    """Ground, light, optional warehouse and one drone at rest at ``test.start_z``. Returns ``(sim, robot)``."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
    from isaaclab.sim.utils import get_prim_at_path, set_prim_visibility
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

    from Drone_RL.uav import uav_cfg as U

    apply_isaac_rtx_global_settings(IsaacRtxRendererGlobalSettingsCfg(
        enable_translucency=True, enable_reflections=True, dlss_mode=2))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device))
    sim.set_camera_view([1.5, 1.5, test.start_z + 0.7], [0.0, 0.0, test.start_z])
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/light", light)
    if warehouse:
        shed = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/{WAREHOUSE_USD}")
        shed.func("/World/warehouse", shed)
        set_prim_visibility(get_prim_at_path("/World/ground/Environment"), False)   # its floor replaces the checker

    cfg = U.UAV_CFG.copy()
    cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=cfg)
    sim.reset()

    pose = robot.data.default_root_pose.torch.clone()
    pose[:, 2] = test.start_z
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=sim.device))
    robot.reset()
    return sim, robot


def read_state(robot) -> SimpleNamespace:
    """What the controller sees: the true state of the simulator (no sensor noise)."""
    data = robot.data
    return SimpleNamespace(pos=data.root_pos_w.torch, vel=data.root_lin_vel_w.torch,
                           quat=data.root_quat_w.torch, rates=data.root_ang_vel_b.torch)


def run(test: LayerTest) -> None:
    """Fly ``test`` in Isaac Sim (parses the command line, so call it from ``__main__``).

    Without arguments the Isaac Sim window and the live plot are on; ``--viz none`` and ``--no_live`` turn them off.
    """
    from isaaclab.app import AppLauncher

    args = parse_args(test)
    simulation_app = AppLauncher(args).app

    # Everything below needs the running app.
    import matplotlib
    matplotlib.use("TkAgg" if args.live else "Agg")
    import matplotlib.pyplot as plt

    from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ
    from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
    from Drone_RL.uav.mdp.actions.propulsion import Propulsion
    from Drone_RL.uav.pid_control.cascade import CascadePID

    dt = 1.0 / FW_TICK_HZ
    realtime = bool(args.visualizer) and "none" not in args.visualizer      # --viz kit: fly at real-time speed
    sim, robot = build_scene(test, dt, args.device, args.warehouse and realtime)    # no warehouse without a window
    body_id = robot.find_bodies("body")[0]
    controller = CascadePID(sim.device)
    propulsion = Propulsion(MotorActionCfg(
        asset_name="robot", use_air_drag=False, use_motor_asymmetry=False, use_thrust_noise=False), 1, sim.device)

    live = None
    if args.live:
        plt.ion()
        live = make_panels(test)
        live[0].tight_layout()

    log_every, plot_every, render_every = (round(s / dt) for s in (LOG_EVERY_S, PLOT_EVERY_S, RENDER_EVERY_S))
    focus = [0.0, 0.0, test.start_z]                 # point the camera looks at, follows the drone with a delay
    wall_start = time.time()
    times, actual, target = [], [], []

    for k in range(int(args.duration / dt)):
        t = k * dt
        wanted = test.setpoint(t)
        state = read_state(robot)

        pwm = test.control(controller, wanted, state, dt)       # layers under test -> PWM of the four motors
        propulsion.step(pwm, robot, body_id)                    # PWM -> wrench on the body
        robot.write_data_to_sim()
        sim.step(render=not realtime)       # window mode renders by hand below, not at every physics step
        robot.update(dt)

        if realtime:
            time.sleep(max(0.0, wall_start + (k + 1) * dt - time.time()))
            if k % render_every == 0:
                focus = [f + CAMERA_SMOOTH * (p - f) for f, p in zip(focus, state.pos[0].tolist())]
                sim.set_camera_view([f + o for f, o in zip(focus, CAMERA_OFFSET)], focus)
                sim.render()

        measured = test.measure(state)
        times.append(t)
        actual.append(measured)
        target.append(wanted)
        if k % log_every == 0:
            log_line(test, t, measured, wanted)
        if live is not None and k % plot_every == 0:
            fig, axes, lines = live
            draw(axes, lines, times, actual, target)
            fig.canvas.draw()
            fig.canvas.flush_events()
        if state.pos[0, 2] < CRASH_Z or not torch.isfinite(state.pos).all():
            print(f"CRASHED at t={t:.2f}s")
            break

    print(f"{test.title}: tracking error over {times[-1]:.1f} s")
    report(test, actual, target)
    save_plot(test, times, actual, target, args.save)
    print(f"Plot: {args.save}")
    simulation_app.close()
