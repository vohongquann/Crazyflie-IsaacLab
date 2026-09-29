"""Drive the Crazyflie USD kinematically along a known trajectory (no physics forces).

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/run_kinematic.py --pattern figure8

Each physics step the root pose and velocity are written from the trajectory (see ``kinematics.py``), so
the drone follows it exactly. Differential flatness gives the thrust and torque the real drone would need;
they are turned into PWM with the ``uav_cfg`` thrust curve and checked against the motor limits, which
tells whether the trajectory is flyable before the PID is asked to track it.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic Crazyflie flight along a known trajectory.")
parser.add_argument("--pattern", choices=("hover", "circle", "figure8"), default="figure8")
parser.add_argument("--duration", type=float, default=14.0)
parser.add_argument("--size", type=float, default=0.5, help="Pattern amplitude / radius [m].")
parser.add_argument("--period", type=float, default=8.0, help="Pattern period [s].")
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
from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ  # noqa: E402
from Drone_RL.uav.mdp.actions.mixer import force_allocation_inverse  # noqa: E402
from Drone_RL.uav.mdp.actions.propulsion import SPIN_VISUAL_SCALE, spin_propellers  # noqa: E402
from Drone_RL.uav.pid_control import kinematics as K  # noqa: E402

RENDER_SETTINGS = IsaacRtxRendererGlobalSettingsCfg(
    enable_translucency=True,   # render glass / transmissive surfaces
    enable_reflections=True,    # render reflections
    dlss_mode=2,                # 0 (Performance), 1 (Balanced), 2 (Quality, the default), 3 (Auto)
)


def main() -> None:
    dt = 1.0 / FW_TICK_HZ
    apply_isaac_rtx_global_settings(RENDER_SETTINGS)   # before the simulation context creates the renderer
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    sim.set_camera_view([1.8, 1.8, 1.4], [0.0, 0.0, 0.4])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
    cfg = U.UAV_CFG.copy()
    cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=cfg)
    sim.reset()
    dev = sim.device

    traj_kw = dict(size=args.size, period=args.period)
    curve_a, curve_b, curve_c = U.CF_THRUST_COEF_G                    # thrust curve: grams = a PWM^2 + b PWM + c
    motor_forces_from_wrench = force_allocation_inverse(dev).double()   # A_F^-1, section 2.2
    spin_sign = torch.tensor(U.CF_MOTOR_SPIN, device=dev)
    max_pwm, steps_outside_limits = 0.0, 0

    for step in range(int(args.duration / dt)):
        t = step * dt
        # Trajectory at the end of this step: pose and velocity to write, wrench needed.
        state = K.state(args.pattern, t + dt, U.DRONE_MASS_TOTAL_KG, U.DRONE_INERTIA_DIAG, **traj_kw)
        pose = torch.cat([state["position"], state["quat"]]).float().to(dev).unsqueeze(0)
        rotation = K.quat_to_rotmat(pose[:, 3:])[0]                       # body -> world
        body_rates_world = rotation @ state["body_rates"].float().to(dev)
        velocity = torch.cat([state["velocity"].float().to(dev), body_rates_world]).unsqueeze(0)

        robot.write_root_pose_to_sim_index(root_pose=pose)
        robot.write_root_velocity_to_sim_index(root_velocity=velocity)
        sim.step()
        robot.update(dt)

        # Would the real motors produce that wrench? (section 4.5)
        wrench = torch.cat([state["thrust"].reshape(1), state["torque"]]).to(dev)
        motor_forces = motor_forces_from_wrench @ wrench                  # F_i [N]
        grams = motor_forces.clamp(min=0) * 4000.0 / K.GRAVITY            # four-motor total for the curve
        pwm = (-curve_b + torch.sqrt((curve_b**2 - 4 * curve_a * (curve_c - grams)).clamp(min=0))) / (2 * curve_a)
        max_pwm = max(max_pwm, pwm.max().item())
        outside = motor_forces.max() > U.CF_F_MAX_N or motor_forces.min() < 0
        steps_outside_limits += int(outside)

        spin_propellers(robot, motor_forces.float().clamp(min=0.0).unsqueeze(0), spin_sign, SPIN_VISUAL_SCALE)

        if step % int(1.0 / dt) == 0:
            tilt_deg = torch.rad2deg(torch.acos(rotation[2, 2])).item()
            print(f"t={t:5.1f}s pos={[round(v, 3) for v in state['position'].tolist()]} tilt={tilt_deg:4.1f}deg "
                  f"F=[{', '.join(f'{f:.3f}' for f in motor_forces.tolist())}] N")
    print(f"RESULT pattern={args.pattern}: max motor PWM {max_pwm:.0f}/{U.CF_PWM_MAX:.0f}, "
          f"steps outside motor limits {steps_outside_limits}")


if __name__ == "__main__":
    main()
    simulation_app.close()
