"""Training curves and the summary chart of the report, from the TensorBoard logs and ``report/metrics/<layer>.json``
(written by ``compare_layers.py``). No simulator.

    python src/Drone_RL/uav/tools/report_figures.py

    report/figures/training.png   per layer: mean reward, tracking error and action noise of the gain network over the
                                  PPO iterations; the tracking error of the tuned PID on the same task as a line
    report/figures/summary.png    per layer: tracking error on random commands and motor chatter, relative to the PID

A run that was resumed (``--resume``) continues an earlier one: the curve of a layer is the newest run of
``logs/rsl_rl/uav_<layer>_gains`` joined with the runs it resumed from. A layer is drawn only when its frozen policy
exists (``rl_control/frozen/gains/<layer>.pt``), so an old run of a layer not trained yet is left out.
"""
import argparse
import functools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
FROZEN = REPO / "src" / "Drone_RL" / "uav" / "rl_control" / "frozen"
LAYERS = ["rate", "attitude", "velocity", "position"]
NAMES = {"pid": "PID", "gains": "RL gains"}
COLORS = {"pid": "#4c72b0", "gains": "#2ca02c"}
ERROR_UNIT = {"rate": "deg/s", "attitude": "deg", "velocity": "m/s", "position": "m"}   # the units of the README table
ERROR_SCALE = {"rate": 57.29578, "attitude": 57.29578, "velocity": 1.0, "position": 1.0}  # logged in rad

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default=str(REPO / "logs" / "rsl_rl"))
parser.add_argument("--out", default=str(REPO / "report"))
args = parser.parse_args()


@functools.cache
def accumulator(run: Path) -> EventAccumulator:
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def scalars(run: Path, tag: str):
    acc = accumulator(run)
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


def refill_end(run: Path, full: float) -> int:
    """First iteration of a resumed run whose episodes are full length again. Right after ``--resume`` every episode
    restarts, so the statistics only count the short episodes that end first: reward and length dip for a few dozen
    iterations although the policy did not change. Those iterations are left out of the curves."""
    s, v = scalars(run, "Train/mean_episode_length")
    return next((a for a, b in zip(s, v) if b >= 0.95 * full), s[-1] + 1 if s else 0)


def curve(runs, tag):
    """One curve over the runs of a chain; where two runs overlap the later one wins."""
    full = max(max(scalars(r, "Train/mean_episode_length")[1], default=0.0) for r in runs)
    steps, values = [], []
    for k, run in enumerate(runs):
        s, v = scalars(run, tag)
        start = refill_end(run, full) if k > 0 else -1
        cut = scalars(runs[k + 1], tag)[0][0] if k + 1 < len(runs) else float("inf")
        kept = [(a, b) for a, b in zip(s, v) if start <= a < cut]
        if steps and kept:
            steps.append(float("nan"))      # a gap where the resume dip was cut out
            values.append(float("nan"))
        steps += [a for a, _ in kept]
        values += [b for _, b in kept]
    return steps, values


def smooth(values, k=15):
    """Moving average; a NaN (the gap of a resume) restarts it."""
    out, acc = [], []
    for v in values:
        acc = [] if v != v else (acc + [v])[-k:]
        out.append(sum(acc) / len(acc) if acc else float("nan"))
    return out


metrics = {}
for layer in LAYERS:
    path = Path(args.out) / "metrics" / f"{layer}.json"
    if path.exists():
        metrics[layer] = json.loads(path.read_text())["methods"]

# ── Training curves ───────────────────────────────────────────────────────────────────────────────────────────
columns = [("Train/mean_reward", "mean episode reward"), ("Metrics/layer/error", "tracking error"),
           ("Policy/mean_std", "action noise (std)")]
fig, ax = plt.subplots(len(LAYERS), 3, figsize=(15, 3.1 * len(LAYERS)))
for i, layer in enumerate(LAYERS):
    runs = chain(Path(args.logs) / f"uav_{layer}_gains") if (FROZEN / "gains" / f"{layer}.pt").exists() else []
    if runs:
        print(f"TRAINING {layer}: {[r.name for r in runs]}")
    for j, (tag, _) in enumerate(columns):
        s, v = curve(runs, tag) if runs else ([], [])
        if j == 1:
            v = [x * ERROR_SCALE[layer] for x in v]
        if s:
            ax[i, j].plot(s, v, color=COLORS["gains"], alpha=0.2, lw=0.7)
            ax[i, j].plot(s, smooth(v), color=COLORS["gains"], lw=1.5, label=NAMES["gains"])
    pid = metrics.get(layer, {}).get("pid", {})
    if pid.get("trained"):
        ax[i, 1].axhline(pid["random"]["error_raw"] * ERROR_SCALE[layer], color=COLORS["pid"], ls="--", lw=1.3,
                         label="PID (same task)")
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
width = 0.35
for j, (key, title) in enumerate((("error", "tracking error on random commands"), ("motor_change", "motor chatter (change per step)"))):
    for k, method in enumerate(("pid", "gains")):
        for i, layer in enumerate(LAYERS):
            m = metrics.get(layer, {})
            x = i + (k - 0.5) * width
            if not m.get(method, {}).get("trained") or not m.get("pid", {}).get("trained"):
                ax[j].text(x, 3, "not trained", ha="center", fontsize=7, color="gray", rotation=90)
                continue
            value = m[method]["random"][key] / m["pid"]["random"][key] * 100.0
            ax[j].bar(x, value, width, color=COLORS[method], label=NAMES[method])
            ax[j].text(x, value, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    ax[j].axhline(100.0, color="k", lw=0.8)
    ax[j].set_xticks(range(len(LAYERS)), LAYERS)
    ax[j].set_ylabel("% of the PID (lower is better)")
    ax[j].set_title(title)
    handles, labels = ax[j].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax[j].legend(unique.values(), unique.keys(), fontsize=8, loc="upper center", ncol=2)
    ax[j].set_ylim(0, 1.18 * max(p.get_height() for p in ax[j].patches))     # room for the legend above the bars
    ax[j].grid(alpha=0.3, axis="y")
fig.suptitle("Each layer as a full cascade from that layer down, relative to the tuned PID cascade")
fig.tight_layout()
fig.savefig(Path(args.out) / "figures" / "summary.png", dpi=110)
print("REPORT figures written to", Path(args.out) / "figures")
