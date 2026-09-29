"""Fly the cascaded PID on the Crazyflie 2.1 Brushless USD and print the tracking error.

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/pid_hover_test.py --scenario square --viz kit

Scenarios:
    hover   hold 0.5 m
    square  hover, then a 0.5 m square at 0.5 m height, 2.5 s per corner

Plant: the same ``Propulsion`` as the RL action (PWM -> thrust curve -> motor lag -> wrench on the body), without
drag, motor gain spread or thrust noise. The controller reads the true simulator state (no sensor noise), so this is
the noise-free baseline to compare the RL tasks against.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cascaded PID flight on the Crazyflie USD.")
parser.add_argument("--scenario", choices=("hover", "square"), default="hover")
parser.add_argument("--duration", type=float, default=12.0, help="Flight time [s].")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motor_tau", type=float, default=None,
                    help="Motor lag while speeding up [s]; default MOTOR_TAU_INC_RANGE[0].")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg  # noqa: E402
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings  # noqa: E402

from Drone_RL.uav import uav_cfg as U  # noqa: E402
from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ, MOTOR_TAU_DEC_RANGE, MOTOR_TAU_INC_RANGE  # noqa: E402
from Drone_RL.uav.mdp.actions.motor_action import MotorActionCfg  # noqa: E402
from Drone_RL.uav.mdp.actions.propulsion import Propulsion  # noqa: E402
from Drone_RL.uav.pid_control.cascade import CascadePID  # noqa: E402

RENDER_SETTINGS = IsaacRtxRendererGlobalSettingsCfg(
    enable_translucency=True,   # render glass / transmissive surfaces
    enable_reflections=True,    # render reflections
    dlss_mode=2,                # 0 (Performance), 1 (Balanced), 2 (Quality, the default), 3 (Auto)
)

HOVER_Z = 0.5          # [m]
SETTLE_S = 3.0         # [s] takeoff time: no error is counted before it, and the square starts after it
ENV_SPACING = 1.0      # [m] between the drones when num_envs > 1


def setpoint(scenario: str, t: float) -> tuple[float, float, float]:
    """Position setpoint (x, y, z) [m] at time ``t`` [s]."""
    if scenario == "hover" or t < SETTLE_S:
        return 0.0, 0.0, HOVER_Z
    corners = [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)]           # square
    x, y = corners[min(int((t - SETTLE_S) // 2.5), len(corners) - 1)]
    return x, y, HOVER_Z


def build_scene(num_envs: int, dt: float) -> tuple[SimulationContext, Articulation]:
    """Ground, light and ``num_envs`` drones side by side; the simulation is reset and ready to step."""
    apply_isaac_rtx_global_settings(RENDER_SETTINGS)   # before the simulation context creates the renderer
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    sim.set_camera_view([1.5, 1.5, 1.2], [0.0, 0.0, 0.4])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=(i * ENV_SPACING, 0.0, 0.0))
    cfg = U.UAV_CFG.copy()
    cfg.prim_path = "/World/Origin.*/Robot"
    robot = Articulation(cfg=cfg)
    sim.reset()
    return sim, robot


def main() -> None:
    dt = 1.0 / FW_TICK_HZ
    n = args.num_envs
    sim, robot = build_scene(n, dt)
    dev = sim.device
    body_id = robot.find_bodies("body")[0]
    origins = torch.tensor([[i * ENV_SPACING, 0.0, 0.0] for i in range(n)], device=dev)

    # Controller: knows the nominal mass, inertia and thrust curve, not the motor lag.
    controller = CascadePID(dev)

    # Plant: fixed motor lag (range of width zero), no drag, no motor gain spread, no thrust noise.
    tau_inc = args.motor_tau if args.motor_tau is not None else MOTOR_TAU_INC_RANGE[0]
    propulsion = Propulsion(MotorActionCfg(
        asset_name="robot", tau_inc_range=(tau_inc, tau_inc), tau_dec_range=MOTOR_TAU_DEC_RANGE,
        use_air_drag=False, use_motor_asymmetry=False, use_thrust_noise=False), n, dev)

    # Start on the ground, at rest.
    pose = robot.data.default_root_pose.torch.clone()
    pose[:, :3] += origins
    robot.write_root_pose_to_sim_index(root_pose=pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev))
    robot.reset()

    sq_err = torch.zeros(n, 3, device=dev)     # sum of the squared position error after SETTLE_S
    counted = 0
    log_every = int(1.0 / dt)                  # print once per simulated second
    for k in range(int(args.duration / dt)):
        t = k * dt
        target = torch.tensor(setpoint(args.scenario, t), device=dev).expand(n, 3) + origins

        # Controller: state -> PWM of the four motors.
        state = robot.data
        pos = state.root_pos_w.torch
        pwm = controller.step(target, pos, state.root_lin_vel_w.torch, state.root_quat_w.torch,
                              state.root_ang_vel_b.torch, dt)

        # Plant: PWM -> wrench on the body, then one physics step.
        propulsion.step(pwm, robot, body_id, dt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        err = pos - target
        if t >= SETTLE_S:
            sq_err += err ** 2
            counted += 1
        if k % log_every == 0:
            e = err[0]
            print(f"t={t:5.1f}s  pos={pos[0].tolist()}  err=({e[0]:+.3f}, {e[1]:+.3f}, {e[2]:+.3f}) m  "
                  f"pwm={pwm[0].mean().item():.0f}")
        if pos[:, 2].min() < -0.05 or not torch.isfinite(pos).all():
            print(f"CRASHED at t={t:.2f}s")
            break

    if counted:
        rms = (sq_err / counted).sqrt().mean(0)
        print(f"RESULT scenario={args.scenario} motor_tau={tau_inc}s  RMS error after {SETTLE_S:g} s (x,y,z) = "
              f"{rms[0]:.3f}, {rms[1]:.3f}, {rms[2]:.3f} m")


if __name__ == "__main__":
    main()
    simulation_app.close()
