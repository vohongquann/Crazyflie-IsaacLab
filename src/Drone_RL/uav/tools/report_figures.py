"""Training curves and the summary chart of the report, from the TensorBoard logs and ``report/metrics/<layer>.json``
(written by ``compare_layers.py``). No simulator.

    python src/Drone_RL/uav/tools/report_figures.py

    report/figures/training.png   per layer: mean reward, tracking error and episode length over the PPO iterations,
                                  RL and RL gains; the tracking error of the tuned PID on the same task as a line
    report/figures/summary.png    per layer: tracking error on random commands and motor chatter, relative to the PID

A run that was resumed (``--resume``) continues an earlier one: the curve of a layer is the newest run of
``logs/rsl_rl/uav_<layer>[_gains]`` joined with the runs it resumed from. A layer is drawn only when its frozen policy
exists (``rl_control/frozen/{rl,gains}/<layer>.pt``), so an old run of a layer not trained yet is left out.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
FROZEN = REPO / "src" / "Drone_RL" / "uav" / "rl_control" / "frozen"
LAYERS = ["rate", "attitude", "velocity", "position"]
NAMES = {"pid": "PID", "rl": "RL", "gains": "RL gains"}
COLORS = {"pid": "#4c72b0", "rl": "#dd5555", "gains": "#2ca02c"}
ERROR_UNIT = {"rate": "rad/s", "attitude": "rad", "velocity": "m/s", "position": "m"}

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default=str(REPO / "logs" / "rsl_rl"))
parser.add_argument("--out", default=str(REPO / "report"))
args = parser.parse_args()


def scalars(run: Path, tag: str):
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [], []
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def chain(experiment: Path):
    """The newest run and the runs it resumed from, oldest first."""
    runs = sorted(r for r in experiment.glob("*/") if list(r.glob("events*")))
    runs = [r for r in runs if scalars(r, "Train/mean_reward")[0]]
    if not runs:
        return []
    picked = [runs.pop()]
    while runs and scalars(picked[0], "Train/mean_reward")[0][0] > 0:
        start = scalars(picked[0], "Train/mean_reward")[0][0]
        earlier = [r for r in runs if scalars(r, "Train/mean_reward")[0][-1] >= start - 1]
        if not earlier:
            break
        picked.insert(0, earlier[-1])
        runs = runs[:runs.index(earlier[-1])]
    return picked


def curve(runs, tag):
    """One curve over the runs of a chain; where two runs overlap the later one wins."""
    steps, values = [], []
    for k, run in enumerate(runs):
        s, v = scalars(run, tag)
        if k + 1 < len(runs):
            cut = scalars(runs[k + 1], tag)[0][0]
            s, v = zip(*[(a, b) for a, b in zip(s, v) if a < cut]) if any(a < cut for a in s) else ([], [])
        steps += list(s)
        values += list(v)
    return steps, values


def smooth(values, k=15):
    out, acc = [], []
    for v in values:
        acc = (acc + [v])[-k:]
        out.append(sum(acc) / len(acc))
    return out


metrics = {}
for layer in LAYERS:
    path = Path(args.out) / "metrics" / f"{layer}.json"
    if path.exists():
        metrics[layer] = json.loads(path.read_text())["methods"]

# ── Training curves ───────────────────────────────────────────────────────────────────────────────────────────
columns = [("Train/mean_reward", "mean episode reward"), ("Metrics/layer/error", "tracking error"),
           ("Train/mean_episode_length", "episode length [steps]")]
fig, ax = plt.subplots(len(LAYERS), 3, figsize=(15, 3.1 * len(LAYERS)))
for i, layer in enumerate(LAYERS):
    for method, suffix in (("rl", ""), ("gains", "_gains")):
        if not (FROZEN / method / f"{layer}.pt").exists():
            continue
        runs = chain(Path(args.logs) / f"uav_{layer}{suffix}")
        if not runs:
            continue
        print(f"TRAINING {layer} {method}: {[r.name for r in runs]}")
        for j, (tag, _) in enumerate(columns):
            s, v = curve(runs, tag)
            if s:
                ax[i, j].plot(s, v, color=COLORS[method], alpha=0.2, lw=0.7)
                ax[i, j].plot(s, smooth(v), color=COLORS[method], lw=1.5, label=NAMES[method])
    pid = metrics.get(layer, {}).get("pid", {})
    if pid.get("trained"):
        ax[i, 1].axhline(pid["random"]["error_raw"], color=COLORS["pid"], ls="--", lw=1.3, label="PID (same task)")
    for j, (tag, title) in enumerate(columns):
        ax[i, j].set_title(f"{layer}: {title}" + (f" [{ERROR_UNIT[layer]}]" if j == 1 else ""), fontsize=10)
        ax[i, j].grid(alpha=0.3)
        if ax[i, j].lines:
            ax[i, j].legend(fontsize=8)
        else:
            ax[i, j].text(0.5, 0.5, "not trained yet", ha="center", va="center", transform=ax[i, j].transAxes, color="gray")
    ax[i, 1].set_yscale("log")
for j in range(3):
    ax[-1, j].set_xlabel("PPO iteration")
fig.suptitle("Training curves (one layer per row, trained bottom-up over the frozen layers below)")
fig.tight_layout()
(Path(args.out) / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(Path(args.out) / "figures" / "training.png", dpi=100)

# ── Summary: relative to the PID ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
width = 0.26
for j, (key, title) in enumerate((("error", "tracking error on random commands"), ("motor_change", "motor chatter (change per step)"))):
    for k, method in enumerate(("pid", "rl", "gains")):
        for i, layer in enumerate(LAYERS):
            m = metrics.get(layer, {})
            x = i + (k - 1) * width
            if not m.get(method, {}).get("trained") or not m.get("pid", {}).get("trained"):
                ax[j].text(x, 12 if key == "motor_change" else 3, "not trained", ha="center", fontsize=7, color="gray", rotation=90)
                continue
            value = m[method]["random"][key] / m["pid"]["random"][key] * 100.0
            ax[j].bar(x, value, width, color=COLORS[method], label=NAMES[method])
            ax[j].text(x, value * 1.04 if key == "motor_change" else value + 2, f"{value:.0f}", ha="center", fontsize=7)
    ax[j].axhline(100.0, color="k", lw=0.8)
    if key == "motor_change":         # the direct RL layers chatter more than ten times the PID: log scale
        ax[j].set_yscale("log")
        ax[j].set_ylim(10, 3000)
    ax[j].set_xticks(range(len(LAYERS)), LAYERS)
    ax[j].set_ylabel("% of the PID (lower is better)")
    ax[j].set_title(title)
    handles, labels = ax[j].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[j].legend(unique.values(), unique.keys(), fontsize=8)
    ax[j].grid(alpha=0.3, axis="y")
fig.suptitle("Each layer as a full cascade from that layer down, relative to the tuned PID cascade")
fig.tight_layout()
fig.savefig(Path(args.out) / "figures" / "summary.png", dpi=110)
print("REPORT figures written to", Path(args.out) / "figures")
