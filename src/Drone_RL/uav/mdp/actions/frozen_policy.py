"""A trained gain layer that no longer learns: the TorchScript actor that ``play`` exports (``exported/policy.pt``),
copied to ``rl_control/frozen/gains/<layer>.pt``.

Pure torch: no Isaac needed.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_frozen_gains(layer: str, device, frozen_dir: str | Path) -> torch.jit.ScriptModule:
    """The frozen gain network of ``layer`` (``<frozen_dir>/<layer>.pt``, ``rl_control/frozen/gains/``: observation -> 9 gains), checked against
    the layer definition of ``mdp/gains.py``."""
    from Drone_RL.uav.mdp.gains import GAIN_LAYERS

    path = Path(frozen_dir) / f"{layer}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen '{layer}' gain layer at {path}. Train it, run play on it and copy the exported policy:\n"
            f"  isaaclab train --rl_library rsl_rl --task Isaac-UAV-{layer.capitalize()}-Gains-v0 ...\n"
            f"  cp logs/rsl_rl/uav_{layer}_gains/<run>/exported/policy.pt {path}")
    policy = torch.jit.load(str(path), map_location=device)
    weights = [v for k, v in policy.state_dict().items() if k.startswith("mlp.") and k.endswith(".weight")]
    obs_dim, action_dim = weights[0].shape[1], weights[-1].shape[0]
    definition = GAIN_LAYERS[layer]
    if (obs_dim, action_dim) != (definition.obs_dim, definition.action_dim):
        raise ValueError(
            f"Frozen '{layer}' gain layer ({path}) has obs/action size {obs_dim}/{action_dim}, but the task now defines "
            f"{definition.obs_dim}/{definition.action_dim}: retrain it, play it and copy the exported policy again.")
    return policy
