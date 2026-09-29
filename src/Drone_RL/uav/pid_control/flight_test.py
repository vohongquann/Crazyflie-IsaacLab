"""Runner shared by the layer tests (rate.py, attitude.py, velocity.py, position.py): fly one layer in Isaac Sim.

Each layer file defines a ``LayerTest`` (setpoint steps, how to drive the layers, what to plot) and calls ``run`` from its
``__main__``. ``run`` starts Isaac Sim, flies the test on the Crazyflie USD with the same ``Propulsion`` as the RL action
(fixed motor lag, no drag, no thrust noise, true simulator state), prints the tracking error and draws actual vs target:

    --viz kit    Isaac Sim window; the flight then runs at real-time speed and the camera follows the drone
    --live       plot redrawn while flying (like the FPV-Drone-Tracking demos)
    --save PATH  PNG written at the end (default ``pid_<layer>.png`` in the current directory)

The module itself has no Isaac import at load time, so the layer files stay importable in tests.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import torch


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


def save_plot(test: LayerTest, times: list, actual: list, target: list, path: str) -> None:
    """PNG with one panel per value: actual (blue) against the setpoint (red dashed)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(test.labels), 1, figsize=(10, 2.8 * len(test.labels)), sharex=True)
    axes = [axes] if len(test.labels) == 1 else list(axes)
    for i, (ax, label) in enumerate(zip(axes, test.labels)):
        ax.plot(times, [row[i] for row in actual], "b-", lw=1.8, label="actual")
        ax.plot(times, [row[i] for row in target], "r--", lw=1.4, label="setpoint")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(test.title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(test: LayerTest, actual: list, target: list) -> None:
    error = torch.tensor(actual) - torch.tensor(target)
    rms = error.pow(2).mean(0).sqrt()
    peak = error.abs().max(0).values
    for label, r, p in zip(test.labels, rms.tolist(), peak.tolist()):
        print(f"  {label:<18} RMS error {r:8.3f}   max |error| {p:8.3f}")


def run(test: LayerTest) -> None:
    """Fly ``test`` in Isaac Sim (parses the command line, so call it from ``__main__``)."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=test.title)
    parser.add_argument("--duration", type=float, default=test.duration, help="Flight time [s].")
    parser.add_argument("--motor_tau", type=float, default=None,
                        help="Motor lag while speeding up [s]; default MOTOR_TAU_INC_RANGE[0].")
    parser.add_argument("--live", action="store_true", help="Redraw the plot while flying.")
    parser.add_argument("--save", type=str, default=f"pid_{test.name}.png", help="PNG written at the end.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationContext
    from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

    import matplotlib
    matplotlib.use("TkAgg" if args.live else "Agg")

    from Drone_RL.uav import uav_cfg as U
    from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ, MOTOR_TAU_DEC_RANGE, MOTOR_TAU_INC_RANGE
    from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg
    from Drone_RL.uav.mdp.actions.propulsion import Propulsion
    from Drone_RL.uav.pid_control.cascade import CascadePID

    dt = 1.0 / FW_TICK_HZ

    # Scene: ground, light, one drone.
    apply_isaac_rtx_global_settings(IsaacRtxRendererGlobalSettingsCfg(
        enable_translucency=True, enable_reflections=True, dlss_mode=2))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    sim.set_camera_view([1.5, 1.5, test.start_z + 0.7], [0.0, 0.0, test.start_z])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    cfg = U.UAV_CFG.copy()
    cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=cfg)
    sim.reset()
    dev = sim.device
    body_id = robot.find_bodies("body")[0]

    controller = CascadePID(dev)
    tau_inc = args.motor_tau if args.motor_tau is not None else MOTOR_TAU_INC_RANGE[0]
    propulsion = Propulsion(MotorActionCfg(
        asset_name="robot", tau_inc_range=(tau_inc, tau_inc), tau_dec_range=MOTOR_TAU_DEC_RANGE,
        use_air_drag=False, use_motor_asymmetry=False, use_thrust_noise=False), 1, dev)

    # Start at rest at ``start_z``.
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, 2] = test.start_z
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev))
    robot.reset()

    live = None
    if args.live:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, axes = plt.subplots(len(test.labels), 1, figsize=(10, 2.8 * len(test.labels)), sharex=True)
        axes = [axes] if len(test.labels) == 1 else list(axes)
        lines = []
        for ax, label in zip(axes, test.labels):
            (la,) = ax.plot([], [], "b-", lw=1.8, label="actual")
            (lt,) = ax.plot([], [], "r--", lw=1.4, label="setpoint")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            lines.append((la, lt))
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(test.title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        live = (fig, axes, lines)

    windowed = bool(args.visualizer) and "none" not in args.visualizer     # --viz kit: fly at real-time speed
    wall_start = time.time()
    times, actual, target = [], [], []
    log_every = int(1.0 / dt)
    plot_every = 10
    for k in range(int(args.duration / dt)):
        t = k * dt
        wanted = test.setpoint(t)
        data = robot.data
        state = SimpleNamespace(pos=data.root_pos_w.torch, vel=data.root_lin_vel_w.torch,
                                quat=data.root_quat_w.torch, rates=data.root_ang_vel_b.torch)

        pwm = test.control(controller, wanted, state, dt)
        propulsion.step(pwm, robot, body_id, dt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if windowed:
            time.sleep(max(0.0, wall_start + (k + 1) * dt - time.time()))
            if k % 10 == 0:
                x, y, z = state.pos[0].tolist()
                sim.set_camera_view([x - 1.2, y - 1.2, z + 0.6], [x, y, z])

        measured = test.measure(state)
        times.append(t)
        actual.append(measured)
        target.append(wanted)
        if k % log_every == 0:
            print(f"t={t:5.1f}s  " + "  ".join(f"{lab.split(' [')[0]}={m:+8.2f} (want {w:+7.2f})"
                                                for lab, m, w in zip(test.labels, measured, wanted)))
        if live is not None and k % plot_every == 0:
            fig, axes, lines = live
            for i, (la, lt) in enumerate(lines):
                la.set_data(times, [row[i] for row in actual])
                lt.set_data(times, [row[i] for row in target])
                axes[i].relim()
                axes[i].autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        if state.pos[0, 2] < -0.05 or not torch.isfinite(state.pos).all():
            print(f"CRASHED at t={t:.2f}s")
            break

    print(f"{test.title}: tracking error over {times[-1]:.1f} s")
    report(test, actual, target)
    save_plot(test, times, actual, target, args.save)
    print(f"Plot: {args.save}")
    simulation_app.close()
