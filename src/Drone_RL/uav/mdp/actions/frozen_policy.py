"""A trained layer that no longer learns: rebuilds the rsl_rl actor from a frozen file (``<frozen_dir>/<layer>.pt``).

The frozen files are written by ``rl_control/freeze.py``. Pure torch: no Isaac needed.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn

from Drone_RL.uav.mdp.layers import LAYERS

NORMALIZER_EPS = 1e-2          # eps of rsl_rl's EmpiricalNormalization: x_norm = (x - mean) / (std + eps)


class FrozenPolicy(nn.Module):
    """Deterministic actor of rsl_rl (``MLPModel``): observation normaliser, then Linear/ELU layers, mean output."""

    def __init__(self, actor_state_dict: dict):
        super().__init__()
        mean = actor_state_dict.get("obs_normalizer._mean")
        std = actor_state_dict.get("obs_normalizer._std")
        self.register_buffer("mean", mean if mean is not None else torch.zeros(1))
        self.register_buffer("std", std if std is not None else torch.ones(1) - NORMALIZER_EPS)

        indices = sorted(int(m.group(1)) for key in actor_state_dict if (m := re.fullmatch(r"mlp\.(\d+)\.weight", key)))
        linears = []
        for i in indices:
            weight, bias = actor_state_dict[f"mlp.{i}.weight"], actor_state_dict[f"mlp.{i}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            linears.append(linear)
        self.linears = nn.ModuleList(linears)
        self.obs_dim = linears[0].in_features
        self.action_dim = linears[-1].out_features
        self.eval()
        self.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / (self.std + NORMALIZER_EPS)
        for linear in self.linears[:-1]:
            x = nn.functional.elu(linear(x))
        return self.linears[-1](x)


def load_frozen(layer: str, device, frozen_dir: str | Path) -> FrozenPolicy:
    """The frozen network of ``layer``; checks that it still matches the layer definition of ``layers.py``."""
    path = Path(frozen_dir) / f"{layer}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"No frozen '{layer}' layer at {path}. Train it and freeze it first:\n"
            f"  isaaclab train --rl_library rsl_rl --task Isaac-UAV-{layer.capitalize()}-RL-v0 ...\n"
            f"  python -m Drone_RL.uav.rl_control.freeze --layer {layer}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    policy = FrozenPolicy(saved["actor_state_dict"]).to(device)
    definition = LAYERS[layer]
    if (policy.obs_dim, policy.action_dim) != (definition.obs_dim, definition.action_dim):
        raise ValueError(
            f"Frozen '{layer}' layer ({path}) has obs/action size {policy.obs_dim}/{policy.action_dim}, but layers.py "
            f"now defines {definition.obs_dim}/{definition.action_dim}: the layer changed, retrain and freeze it again.")
    return policy
