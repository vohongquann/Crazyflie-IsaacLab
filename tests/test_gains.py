"""Tests of the gain tasks that need no simulator: the map from the network output to PID gains, the controllers with
gains per environment, and the task configs."""

import pytest
import torch

from Drone_RL.uav.mdp.gains import FACTOR, GAIN_LAYERS, KD_PER_KP, KI_PER_KP
from Drone_RL.uav.pid_control.attitude import AttitudeController
from Drone_RL.uav.pid_control.position import PositionController
from Drone_RL.uav.pid_control.rate import RateController
from Drone_RL.uav.pid_control.velocity import VelocityController

N = 5


@pytest.mark.parametrize("name", list(GAIN_LAYERS))
def test_zero_action_is_the_tuned_pid(name):
    """Nonzero gains of the tuned PID come back as they are; the gains it does not have are off."""
    layer = GAIN_LAYERS[name]
    gains = torch.stack(layer.gains(torch.zeros(N, 9)), dim=1)             # (N, kind, axis)
    expected = torch.where(layer.nominal > 0.0, layer.nominal, torch.zeros(3, 3)).expand(N, 3, 3)
    assert torch.allclose(gains, expected)


@pytest.mark.parametrize("name", list(GAIN_LAYERS))
def test_action_limits(name):
    layer = GAIN_LAYERS[name]
    low = torch.stack(layer.gains(-torch.ones(N, 9)), dim=1)[0]
    high = torch.stack(layer.gains(torch.ones(N, 9)), dim=1)[0]
    on = layer.nominal > 0.0
    assert torch.allclose(low[on], layer.nominal[on] / FACTOR)
    assert torch.allclose(high[on], layer.nominal[on] * FACTOR)
    assert torch.all(low[~on] == 0.0)
    assert torch.allclose(high[~on], layer.maximum[~on])
    kp = layer.nominal[0]
    if layer.nominal[1].eq(0.0).all():
        assert torch.allclose(high[1], KI_PER_KP * kp)
    if layer.nominal[2].eq(0.0).all():
        assert torch.allclose(high[2], KD_PER_KP * kp)


def test_action_is_clamped_and_ordered_by_kind():
    layer = GAIN_LAYERS["velocity"]
    action = torch.zeros(1, 9)
    action[0, 4] = 100.0                      # ki of y, far beyond 1
    kp, ki, kd = layer.gains(action)
    assert ki[0, 1] == pytest.approx(layer.nominal[1, 1].item() * FACTOR)
    assert torch.allclose(kp[0], layer.nominal[0]) and torch.allclose(ki[0, [0, 2]], layer.nominal[1, [0, 2]])


def _gains_per_env(name):
    layer = GAIN_LAYERS[name]
    kp, ki, kd = layer.gains(torch.zeros(N, 9))
    return layer, kp, ki, kd


@pytest.mark.parametrize("controller_class, name", [
    (RateController, "rate"), (AttitudeController, "attitude"),
    (VelocityController, "velocity"), (PositionController, "position"),
])
def test_tuned_gains_per_environment_change_nothing(controller_class, name):
    """The PID with the tuned gains given as (N, 3) tensors and an integral limit gives the output of the default one."""
    torch.manual_seed(0)
    layer, kp, ki, kd = _gains_per_env(name)
    default, given = controller_class("cpu"), controller_class("cpu")
    given.pid.kp, given.pid.ki, given.pid.kd = kp, ki, kd
    given.pid.int_limit = layer.int_limit
    for _ in range(20):
        error = 0.1 * torch.randn(N, 3)
        assert torch.allclose(given.pid.update(error, 0.01), default.pid.update(error, 0.01), atol=1e-5)


def test_gains_differ_per_environment():
    """A different kp in each environment gives a different output in each environment."""
    controller = PositionController("cpu")
    controller.pid.kp = torch.tensor([[1.0] * 3, [2.0] * 3])
    out = controller.update(0.1 * torch.ones(2, 3), torch.zeros(2, 3), 0.02)     # below the output limit
    assert torch.allclose(out[1], 2.0 * out[0])


@pytest.mark.parametrize("task, layer", [
    ("Isaac-UAV-Rate-Gains-v0", "rate"), ("Isaac-UAV-Attitude-Gains-v0", "attitude"),
    ("Isaac-UAV-Velocity-Gains-v0", "velocity"), ("Isaac-UAV-Position-Gains-v0", "position"),
])
def test_task_configs(task, layer):
    import importlib

    import gymnasium as gym

    import Drone_RL.tasks  # noqa: F401  registers the tasks
    from Drone_RL.uav.mdp.actions.gain_action import GainCascadeActionCfg

    spec = gym.spec(task)
    module, name = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(module), name)()
    assert isinstance(cfg.actions.cascade, GainCascadeActionCfg)
    assert cfg.actions.cascade.layer == layer and cfg.commands.layer.layer == layer
    module, name = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent = getattr(importlib.import_module(module), name)()
    assert agent.experiment_name == f"uav_{layer}_gains"


def _export(tmp_path, layer, action_dim=9):
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    obs_dim = GAIN_LAYERS[layer].obs_dim
    obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    model = MLPModel(obs, {"actor": ["policy"]}, "actor", action_dim, hidden_dims=[64, 64], obs_normalization=True)
    model.eval()
    torch.jit.script(model.as_jit()).save(str(tmp_path / f"{layer}.pt"))
    return model


def test_observation_sizes():
    assert [GAIN_LAYERS[n].obs_dim for n in ("rate", "attitude", "velocity", "position")] == [25, 28, 27, 27]


def test_load_frozen_gains_reads_the_exported_policy(tmp_path):
    from Drone_RL.uav.mdp.actions.frozen_policy import load_frozen_gains

    model = _export(tmp_path, "attitude")
    policy = load_frozen_gains("attitude", "cpu", tmp_path)
    obs = torch.randn(4, GAIN_LAYERS["attitude"].obs_dim)
    from tensordict import TensorDict

    with torch.no_grad():
        assert torch.allclose(policy(obs), model(TensorDict({"policy": obs}, batch_size=[4])), atol=1e-5)
    with pytest.raises(FileNotFoundError):
        load_frozen_gains("rate", "cpu", tmp_path)
    _export(tmp_path, "velocity", action_dim=3)           # a policy of the old kind: 3 outputs
    with pytest.raises(ValueError):
        load_frozen_gains("velocity", "cpu", tmp_path)
