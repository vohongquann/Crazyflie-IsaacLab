"""Compare the two controllers of one layer: tuned PID and RL gains (the network writes the 9 PID gains of the layer).
Each one is a full cascade from that layer down:

    pid       tuned PID of the layer over the tuned PID layers below            (Isaac-UAV-<Layer>-Gains-v0, action 0)
    gains     frozen/gains/<layer>.pt over frozen/gains/<layers below>.pt       (Isaac-UAV-<Layer>-Gains-v0)

Two measurements, headless, the same seed for both:

    random    the training task: 64 drones, random commands of the layers above, one episode; mean tracking error of the
              layer, share of drones still flying at the end, motor command change per step (chatter)
    steps     a fixed test signal of the layer (rate and attitude: square steps; velocity: steps; position: target jumps,
              then a circle); mean |error| per axis, plotted

    python src/Drone_RL/uav/tools/compare_layers.py --layer attitude

Writes ``report/figures/compare_<layer>.png`` and ``report/metrics/<layer>.json``. If a frozen gain network is missing
(a layer not trained yet), the gains are skipped and marked so in the json.
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[4]

parser = argparse.ArgumentParser()
parser.add_argument("--layer", required=True, choices=["rate", "attitude", "velocity", "position"])
parser.add_argument("--out", default=str(REPO / "report"), help="report folder (figures/ and metrics/ inside)")
parser.add_argument("--num_envs", type=int, default=8, help="drones of the step test")
parser.add_argument("--eval_envs", type=int, default=64, help="drones of the random-command test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp as isaac_mdp  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from Drone_RL.uav import uav_cfg as U  # noqa: E402
from Drone_RL.uav.mdp.flight import GRAVITY  # noqa: E402
from Drone_RL.uav.mdp.layers import layers_below  # noqa: E402
from Drone_RL.uav.rl_control import gains_env_cfg as G  # noqa: E402
from Drone_RL.uav.rl_control.cascade_env_cfg import FROZEN_GAINS_DIR  # noqa: E402

torch.set_grad_enabled(False)
LAYER = args.layer
METHODS = ["pid", "gains"]
NAMES = {"pid": "PID", "gains": "RL gains"}
COLORS = {"pid": "#4c72b0", "gains": "#2ca02c"}
GAINS_CFG = {"rate": G.RateGainsEnvCfg, "attitude": G.AttitudeGainsEnvCfg, "velocity": G.VelocityGainsEnvCfg,
             "position": G.PositionGainsEnvCfg}
HEIGHT = 1.5

# ── Test signals: (start s, end s, command index, value) ──────────────────────────────────────────────────────
if LAYER == "rate":            # [wx, wy, wz, thrust] rad/s: square waves that bring the drone back to level
    STEPS = [(1.0, 1.5, 0, 0.52), (1.5, 2.0, 0, -0.52), (3.0, 3.5, 1, 0.52), (3.5, 4.0, 1, -0.52),
             (5.0, 5.5, 2, 1.57), (5.5, 6.0, 2, -1.57)]
    T, LABELS, UNIT, SCALE = 7.0, ["roll rate", "pitch rate", "yaw rate"], "deg/s", 180.0 / math.pi
elif LAYER == "attitude":      # [roll, pitch, yaw, thrust] rad
    STEPS = [(0.5, 1.5, 0, 0.2), (2.0, 3.0, 1, -0.2), (3.5, 5.0, 2, 0.8)]
    T, LABELS, UNIT, SCALE = 5.5, ["roll", "pitch", "yaw"], "deg", 180.0 / math.pi
elif LAYER == "velocity":      # [vx, vy, vz, yaw] m/s
    STEPS = [(0.5, 2.5, 0, 1.0), (3.0, 5.0, 1, 1.0), (5.5, 7.0, 2, 0.5), (7.5, 9.5, 0, -1.0), (7.5, 9.5, 1, -1.0)]
    T, LABELS, UNIT, SCALE = 10.0, ["vx", "vy", "vz"], "m/s", 1.0
else:                          # [x, y, z, yaw, target velocity (3)]: jumps, then a circle (radius 0.5 m, 0.5 m/s)
    JUMPS = [(0.0, 0.0, 0.0, HEIGHT), (0.5, 0.5, 0.0, HEIGHT), (3.0, 0.5, 0.5, HEIGHT), (5.5, 0.0, 0.5, HEIGHT + 0.3),
             (8.0, 0.0, 0.0, HEIGHT)]
    CIRCLE_START, RADIUS, SPEED = 10.5, 0.5, 0.5
    T, LABELS, UNIT, SCALE = 10.5 + 2.0 * math.pi * RADIUS / SPEED, ["x", "y", "z"], "m", 1.0


def height_hold_thrust(env):
    """Total thrust [N] that holds 1.5 m (the rate and attitude commands carry a thrust)."""
    z, vz = isaac_mdp.root_pos_w(env)[:, 2], isaac_mdp.root_lin_vel_w(env)[:, 2]
    return U.DRONE_MASS_TOTAL_KG * (GRAVITY + (6.0 * (HEIGHT - z) - 4.0 * vz).clamp(-3.0, 3.0))


def command_at(t, env):
    """Command of the layer at time t (relative to the environment origins)."""
    n, device = env.num_envs, env.device
    if LAYER == "position":
        if t < CIRCLE_START:
            target = JUMPS[0][1:]
            for t0, x, y, z in JUMPS:
                if t >= t0:
                    target = (x, y, z)
            c = [*target, 0.0, 0.0, 0.0, 0.0]
        else:
            phase = SPEED / RADIUS * (t - CIRCLE_START)
            c = [-RADIUS + RADIUS * math.cos(phase), RADIUS * math.sin(phase), HEIGHT, 0.0,
                 -SPEED * math.sin(phase), SPEED * math.cos(phase), 0.0]
        command = torch.tensor(c, device=device).repeat(n, 1)
        command[:, :3] += env.scene.env_origins
        return command
    c = torch.zeros(n, 4, device=device)
    for t0, t1, i, value in STEPS:
        if t0 <= t < t1:
            c[:, i] = value
    if LAYER in ("rate", "attitude"):
        c[:, 3] = height_hold_thrust(env)
        if LAYER == "attitude":       # the vertical part of the thrust stays the weight at the wanted tilt
            c[:, 3] /= torch.cos(c[:, 0]) * torch.cos(c[:, 1])
    return c


def measured(env):
    if LAYER == "rate":
        return isaac_mdp.base_ang_vel(env)
    if LAYER == "attitude":
        return torch.stack(euler_xyz_from_quat(isaac_mdp.root_quat_w(env)), dim=-1)
    if LAYER == "velocity":
        return isaac_mdp.root_lin_vel_w(env)
    return isaac_mdp.root_pos_w(env) - env.scene.env_origins


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def policy_file(layer):
    return FROZEN_GAINS_DIR / f"{layer}.pt"


def missing(method):
    """Frozen files the method needs that do not exist (the layer and every layer below)."""
    if method == "pid":
        return []
    names = [LAYER] + [layer.name for layer in layers_below(LAYER)]
    return [str(policy_file(name).relative_to(REPO)) for name in names if not policy_file(name).exists()]


def make_env(method, test):
    cfg = GAINS_CFG[LAYER]()
    if method == "pid":
        cfg.actions.cascade.pid_below = True
    cfg.scene.num_envs = args.num_envs if test else args.eval_envs
    if test:
        cfg.episode_length_s = T + 5.0
        cfg.events.reset_drone.params["pose_range"] = {}
        cfg.events.reset_drone.params["velocity_range"] = {}
        for name in ("time_out", "too_low", "tilted", "left_volume"):
            setattr(cfg.terminations, name, None)
    return ManagerBasedRLEnv(cfg)


def actor(method, env):
    if method == "pid":
        zeros = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        return lambda obs: zeros
    policy = torch.jit.load(str(policy_file(LAYER)), map_location=env.device)
    return lambda obs: policy(obs).clamp(-1.0, 1.0)


def random_test(method):
    """The training task, one episode: mean layer error, drones still flying, motor change per step."""
    env = make_env(method, test=False)
    torch.manual_seed(0)
    obs, _ = env.reset()
    act = actor(method, env)
    term = env.command_manager.get_term("layer")
    motors = env.action_manager.get_term("cascade")
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    steps = int(env.max_episode_length)
    change, last = 0.0, motors.motor_commands.clone()
    for k in range(steps - 1):
        obs, _, terminated, _, _ = env.step(act(obs["policy"]))
        alive &= ~terminated
        change += (motors.motor_commands - last).abs().mean().item()
        last = motors.motor_commands.clone()
        error = term.metrics["error"].clone()
    env.close()
    scale = 180.0 / math.pi if LAYER in ("rate", "attitude") else 1.0
    return {"error": error.mean().item() * scale, "error_raw": error.mean().item(), "error_unit": "deg/s" if LAYER == "rate" else "deg" if LAYER == "attitude" else UNIT,
            "flying": alive.float().mean().item(), "motor_change": change / (steps - 1)}


def step_test(method):
    env = make_env(method, test=True)
    torch.manual_seed(0)
    env.reset()
    term = env.command_manager.get_term("layer")
    term._update_command = lambda: None              # the command is the test signal
    motors = env.action_manager.get_term("cascade")
    act = actor(method, env)
    log = {k: [] for k in ("t", "cmd", "meas", "motor")}
    for k in range(int(T / env.step_dt)):
        t = k * env.step_dt
        term._command[:] = command_at(t, env)
        obs = env.observation_manager.compute()["policy"]
        env.step(act(obs))
        cmd = term._command[0, :3].clone()
        if LAYER == "position":
            cmd -= env.scene.env_origins[0]
        log["t"].append(t)
        log["cmd"].append(cmd.cpu())
        log["meas"].append(measured(env).cpu().clone())
        log["motor"].append(motors.motor_commands[0, 0].item())
    env.close()
    r = {k: (torch.tensor(v) if k in ("t", "motor") else torch.stack(v)) for k, v in log.items()}
    error = r["cmd"][:, None, :] - r["meas"]                                   # (time, envs, 3)
    if LAYER == "attitude":
        error = wrap(error)
    r["error"] = (error.abs() * SCALE).mean(dim=(0, 1))
    return r


# ── Run ───────────────────────────────────────────────────────────────────────────────────────────────────────
results, metrics = {}, {"layer": LAYER, "unit": UNIT, "methods": {}}
for method in METHODS:
    lacking = missing(method)
    if lacking:
        print(f"COMPARE {LAYER} {method}: skipped, missing {lacking}", flush=True)
        metrics["methods"][method] = {"trained": False, "missing": lacking}
        continue
    random_metrics = random_test(method)
    results[method] = step_test(method)
    metrics["methods"][method] = {"trained": True, "random": random_metrics,
                                  "step_error": [round(v, 4) for v in results[method]["error"].tolist()],
                                  "step_error_mean": round(results[method]["error"].mean().item(), 4)}
    print(f"COMPARE {LAYER} {method}: {json.dumps(metrics['methods'][method])}", flush=True)

out = Path(args.out)
(out / "metrics").mkdir(parents=True, exist_ok=True)
(out / "figures").mkdir(parents=True, exist_ok=True)
json_path = out / "metrics" / f"{LAYER}.json"
json_path.write_text(json.dumps(metrics, indent=2))

# ── Figure: both methods on the same axes ────────────────────────────────────────────────────────────────────
circle = LAYER == "position"
rows = 4
fig = plt.figure(figsize=(13, 2.6 * rows + (0.5 if circle else 0)))
grid = fig.add_gridspec(rows, 3 if circle else 1)
first = next(iter(results.values()))
t = first["t"]
for i in range(3):
    a = fig.add_subplot(grid[i, :2] if circle else grid[i, 0])
    a.plot(t, first["cmd"][:, i] * SCALE, "k--", lw=1.2, label="wanted")
    for method, r in results.items():
        a.plot(t, r["meas"][:, :, i].mean(1) * SCALE, color=COLORS[method], lw=1.4,
               label=f"{NAMES[method]} (mean |error| {r['error'][i]:.3g} {UNIT})")
    a.set_ylabel(f"{LABELS[i]} [{UNIT}]")
    a.legend(fontsize=8, loc="upper right")
    a.grid(alpha=0.3)
a = fig.add_subplot(grid[3, :2] if circle else grid[3, 0])
for method, r in results.items():
    a.plot(t, r["motor"], color=COLORS[method], lw=0.9, label=NAMES[method])
a.set_ylabel("motor 1 command")
a.set_xlabel("t [s]")
a.legend(fontsize=8, loc="upper right")
a.grid(alpha=0.3)
if circle:
    a = fig.add_subplot(grid[:, 2])
    on_circle = t >= CIRCLE_START
    a.plot(first["cmd"][on_circle, 0], first["cmd"][on_circle, 1], "k--", lw=1.2, label="wanted")
    for method, r in results.items():
        xy = r["meas"][on_circle][:, :, :2].mean(1)
        a.plot(xy[:, 0], xy[:, 1], color=COLORS[method], lw=1.4, label=NAMES[method])
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("circle, 0.5 m radius, 0.5 m/s")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
skipped = [NAMES[m] for m in METHODS if m not in results]
fig.suptitle(f"{LAYER} layer: step response of the controllers (mean of {args.num_envs} drones)"
             + (f"; not trained yet: {', '.join(skipped)}" if skipped else ""))
fig.tight_layout()
figure = out / "figures" / f"compare_{LAYER}.png"
fig.savefig(figure, dpi=110)
print("COMPARE saved", figure)
app.close()
