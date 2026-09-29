"""Tests of the RL cascade pieces that need no simulator: layer shapes and frozen networks."""

import torch

from rsl_rl.modules import MLP, EmpiricalNormalization

from Drone_RL.uav.mdp.actions.frozen_policy import FrozenPolicy, load_frozen
from Drone_RL.uav.mdp.layers import LAYERS, FlightState, layers_below
from Drone_RL.uav.rl_control.freeze import freeze


def _state(n: int) -> FlightState:
    return FlightState(torch.randn(n, 3), torch.eye(3).repeat(n, 1, 1), torch.randn(n, 3), torch.randn(n, 3))


def test_every_layer_feeds_the_one_below():
    for layer in LAYERS.values():
        n = 5
        obs = layer.observe(_state(n), torch.randn(n, layer.command_dim), torch.zeros(n, layer.history, layer.action_dim))
        assert obs.shape == (n, layer.obs_dim)
        out = layer.output(torch.rand(n, layer.action_dim) * 2 - 1, torch.randn(n, layer.command_dim), _state(n))
        expected = LAYERS[layer.below].command_dim if layer.below else 4
        assert out.shape == (n, expected)
    assert [layer.name for layer in layers_below("position")] == ["velocity", "attitude", "rate"]


def test_zero_rate_action_at_hover_thrust_is_hover_throttle():
    from Drone_RL.uav import uav_cfg as U
    from Drone_RL.uav.mdp.layers import WEIGHT_N

    command = torch.tensor([[0.0, 0.0, 0.0, WEIGHT_N]])
    motors = LAYERS["rate"].output(torch.zeros(1, 4), command, _state(1))
    assert torch.allclose(motors, torch.full((1, 4), U.DRONE_HOVER_THROTTLE), atol=2e-3)


def _actor_state_dict(obs_dim: int, action_dim: int) -> tuple[dict, torch.nn.Module, torch.nn.Module]:
    """State dict laid out like rsl_rl's MLPModel (normaliser + MLP)."""
    normalizer = EmpiricalNormalization(obs_dim)
    normalizer.update(torch.randn(256, obs_dim) * 3.0 + 1.0)
    mlp = MLP(obs_dim, action_dim, [64, 64], "elu")
    state = {f"obs_normalizer.{k}": v for k, v in normalizer.state_dict().items()}
    state.update({f"mlp.{k}": v for k, v in mlp.state_dict().items()})
    return state, normalizer.eval(), mlp


def test_frozen_policy_matches_rsl_rl_actor():
    state, normalizer, mlp = _actor_state_dict(23, 4)
    obs = torch.randn(10, 23)
    with torch.no_grad():
        expected = mlp(normalizer(obs))
    assert torch.allclose(FrozenPolicy(state)(obs), expected, atol=1e-5)


def test_freeze_writes_a_loadable_layer(tmp_path):
    layer = LAYERS["rate"]
    state, _, _ = _actor_state_dict(layer.obs_dim, layer.action_dim)
    checkpoint = tmp_path / "model_10.pt"
    torch.save({"actor_state_dict": state}, checkpoint)
    freeze("rate", checkpoint, tmp_path / "frozen")
    policy = load_frozen("rate", "cpu", tmp_path / "frozen")
    assert policy(torch.zeros(1, layer.obs_dim)).shape == (1, 4)


def _tilted_state(n: int) -> FlightState:
    from isaaclab.utils.math import matrix_from_quat

    quat = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    return FlightState(torch.randn(n, 3), matrix_from_quat(quat), torch.randn(n, 3), torch.randn(n, 3))


def test_attitude_layer_frames_with_a_tilted_drone():
    from Drone_RL.uav import uav_cfg as U
    from Drone_RL.uav.mdp.layers import GRAVITY

    layer, n = LAYERS["attitude"], 6
    state = _tilted_state(n)
    command = torch.randn(n, 4)
    force_per_mass = command[:, :3] + torch.tensor([0.0, 0.0, GRAVITY])
    obs = layer.observe(state, command, torch.zeros(n, layer.history, layer.action_dim))
    rotation_t = state.rotation.transpose(1, 2)
    assert torch.allclose(obs[:, :3], (rotation_t @ force_per_mass.unsqueeze(-1)).squeeze(-1) / GRAVITY, atol=1e-5)
    world_up_in_body = (rotation_t @ torch.tensor([0.0, 0.0, 1.0])).squeeze(-1)
    assert torch.allclose(obs[:, 4:7], world_up_in_body, atol=1e-5)
    thrust = layer.output(torch.zeros(n, 4), command, state)[:, 3]
    body_up_in_world = state.rotation[:, :, 2]
    expected = (U.DRONE_MASS_TOTAL_KG * (force_per_mass * body_up_in_world).sum(-1)).clamp(min=0.0)
    assert torch.allclose(thrust, expected, atol=1e-5)


def test_frozen_layer_sees_what_the_layer_saw_in_training(monkeypatch):
    """Training: CascadeAction pushes the clipped action, then layer_observation observes. Frozen: FrozenLayer.step."""
    from Drone_RL.uav.mdp.actions import cascade_action
    from Drone_RL.uav.mdp.actions.cascade_action import FrozenLayer, History

    for layer in LAYERS.values():
        n = 3
        weights = torch.randn(layer.obs_dim, layer.action_dim)
        seen = []

        def policy(obs, weights=weights, seen=seen):
            seen.append(obs.clone())
            return 2.0 * obs @ weights                 # large, so the clip matters

        monkeypatch.setattr(cascade_action, "load_frozen", lambda *args, policy=policy: policy)
        frozen = FrozenLayer(layer, n, 500.0, "unused", "cpu")
        history = History(n, layer, "cpu")
        for _ in range(5):
            state, command = _tilted_state(n), torch.randn(n, layer.command_dim)
            expected_obs = layer.observe(state, command, history.values)
            action = (2.0 * expected_obs @ weights).clamp(-1.0, 1.0)
            history.push(action)
            frozen.step(state, command)
            assert torch.allclose(seen[-1], expected_obs)
            assert torch.allclose(frozen.output, layer.output(action, command, state))


def test_cascade_runs_frozen_layers_top_down_at_their_rates():
    from types import SimpleNamespace

    from Drone_RL.uav.mdp.actions.cascade_action import CascadeAction

    calls = []

    class Fake:
        def __init__(self, name, period, width):
            self.name, self.period, self.output = name, period, torch.zeros(1, width)

        def step(self, state, command):
            calls.append((action._tick, self.name, command.clone()))
            self.output = torch.full_like(self.output, float(len(calls)))

    action = CascadeAction.__new__(CascadeAction)
    action.frozen = [Fake("velocity", 10, 4), Fake("attitude", 10, 4), Fake("rate", 5, 4)]
    action._output = torch.full((1, 4), -1.0)
    action._motor = torch.zeros(1, 4)
    action._propulsion = SimpleNamespace(step=lambda *args: None)
    action.cfg = SimpleNamespace(pwm_max=1.0)
    action._robot, action._body_id, action._physics_dt = None, 0, 0.002
    action.state = lambda: None
    action._tick = 0
    for _ in range(10):
        action.apply_actions()

    assert [(tick, name) for tick, name, _ in calls] == [(0, "velocity"), (0, "attitude"), (0, "rate"), (5, "rate")]
    assert torch.equal(calls[0][2], torch.full((1, 4), -1.0))       # velocity gets the trained layer's output
    assert torch.equal(calls[1][2], torch.full((1, 4), 1.0))        # attitude gets the velocity output
    assert torch.equal(calls[3][2], torch.full((1, 4), 2.0))        # rate at tick 5: attitude output held
    assert torch.equal(action._motor, torch.full((1, 4), 4.0))      # motors: the last rate output
