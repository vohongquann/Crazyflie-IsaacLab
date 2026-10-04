"""A trained layer that no longer learns: the TorchScript actor that ``play`` exports (``exported/policy.pt``), copied to ``<frozen_dir>/<layer>.pt``.

Pure torch: no Isaac needed.
"""
from __future__ import annotations

from pathlib import Path

import torch

from Drone_RL.uav.mdp.layers import LAYERS


def load_frozen(layer: str, device, frozen_dir: str | Path) -> torch.jit.ScriptModule:
    """The frozen network of ``layer`` (observation -> action, normaliser inside); checks that it still matches the layer
    definition of ``layers.py``."""
    path = Path(frozen_dir) / f"{layer}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen '{layer}' layer at {path}. Train it, run play on it and copy the exported policy:\n"
            f"  isaaclab train --rl_library rsl_rl --task Isaac-UAV-{layer.capitalize()}-RL-v0 ...\n"
            f"  cp logs/rsl_rl/uav_{layer}/<run>/exported/policy.pt {path}")
    policy = torch.jit.load(str(path), map_location=device)
    weights = [v for k, v in policy.state_dict().items() if k.startswith("mlp.") and k.endswith(".weight")]
    obs_dim, action_dim = weights[0].shape[1], weights[-1].shape[0]
    definition = LAYERS[layer]
    if (obs_dim, action_dim) != (definition.obs_dim, definition.action_dim):
        raise ValueError(
            f"Frozen '{layer}' layer ({path}) has obs/action size {obs_dim}/{action_dim}, but layers.py "
            f"now defines {definition.obs_dim}/{definition.action_dim}: the layer changed, retrain it, play it and copy the exported policy again.")
    return policy
