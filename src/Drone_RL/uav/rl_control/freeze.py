"""Freeze a trained layer: copy the actor of its last checkpoint into ``rl_control/frozen/<layer>.pt``.

The layers above load the frozen file, never the training logs, so retraining a layer does not change the layers built
on it until it is frozen again. Pure torch: no Isaac needed.

    ~/miniconda3/envs/env_isaaclab/bin/python -m Drone_RL.uav.rl_control.freeze --layer rate
    ~/miniconda3/envs/env_isaaclab/bin/python -m Drone_RL.uav.rl_control.freeze --layer rate --checkpoint logs/rsl_rl/uav_rate/<run>/model_499.pt

Without ``--checkpoint`` it takes the newest run of ``logs/rsl_rl/uav_<layer>/`` and its highest ``model_<i>.pt``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from Drone_RL.uav.mdp.actions.frozen_policy import load_frozen
from Drone_RL.uav.mdp.layers import LAYERS

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
"""Frozen weights of the four layers; ``cascade_env_cfg.py`` points the cascade action here."""


def experiment_name(layer: str) -> str:
    return f"uav_{layer}"


def newest_checkpoint(layer: str, log_root: Path) -> Path:
    runs = sorted(p for p in (log_root / experiment_name(layer)).glob("*") if p.is_dir())
    for run in reversed(runs):
        models = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if models:
            return models[-1]
    raise FileNotFoundError(f"No checkpoint under {log_root / experiment_name(layer)}")


def freeze(layer: str, checkpoint: Path, frozen_dir: Path = FROZEN_DIR) -> Path:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = {k: v for k, v in saved["actor_state_dict"].items() if k.startswith(("obs_normalizer.", "mlp."))}
    frozen_dir.mkdir(parents=True, exist_ok=True)
    path = frozen_dir / f"{layer}.pt"
    torch.save({"layer": layer, "actor_state_dict": actor, "source": str(checkpoint)}, path)
    load_frozen(layer, "cpu", frozen_dir)          # fails now rather than in the next training run
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", required=True, choices=list(LAYERS))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--log_root", type=Path, default=Path("logs/rsl_rl"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or newest_checkpoint(args.layer, args.log_root)
    path = freeze(args.layer, checkpoint)
    print(f"Frozen '{args.layer}' layer: {checkpoint} -> {path}")


if __name__ == "__main__":
    main()
