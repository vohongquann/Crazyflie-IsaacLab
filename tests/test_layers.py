"""Tests of the cascade pieces that need no simulator: layer shapes, observation terms and the cascade order."""

from types import SimpleNamespace

import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, wrap_to_pi

from Drone_RL.uav.mdp.actions.constants import FW_TICK_HZ
from Drone_RL.uav.mdp.actions.gain_action import GainCascadeAction
from Drone_RL.uav.mdp.flight import WEIGHT_N
from Drone_RL.uav.mdp.gains import GAIN_LAYERS
from Drone_RL.uav.mdp.layers import LAYERS, layers_below, output_dim

LEVEL = [[0.0, 0.0, 0.0, 1.0]]      # quaternion (x, y, z, w) of a drone that is not tilted


class _Scene:
    def __init__(self, robot, n: int):
        self.robot, self.env_origins = robot, torch.zeros(n, 3)

    def __getitem__(self, name):
        return self.robot


class _FakeEnv:
    """The parts of the environment that the observation terms read: the robot data and the layer being observed."""

    device = "cpu"

    def __init__(self, n: int, tilted: bool = True):
        quat = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1) if tilted else torch.tensor(LEVEL).repeat(n, 1)
        values = {
            "root_pos_w": torch.randn(n, 3),
            "root_quat_w": quat,
            "root_lin_vel_w": torch.randn(n, 3),
            "root_ang_vel_b": torch.randn(n, 3),
            "projected_gravity_b": quat_apply_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)),
        }
        robot = SimpleNamespace(data=SimpleNamespace(**{k: SimpleNamespace(torch=v) for k, v in values.items()}))
        self.n, self.values = n, values
        self.scene = _Scene(robot, n)
        self.cascade = SimpleNamespace(observing=None)
        self.action_manager = SimpleNamespace(get_term=lambda name: self.cascade)

    def observe_as(self, layer, command=None, history=None):
        """The layer in training: its command and its last outputs."""
        command = torch.randn(self.n, layer.command_dim) if command is None else command
        gains = GAIN_LAYERS[layer.name]
        history = torch.randn(self.n, gains.history, gains.action_dim) if history is None else history
        self.cascade.observing = SimpleNamespace(command=command, history=SimpleNamespace(values=history))
        return self


def test_every_layer_feeds_the_one_below():
    for layer in LAYERS.values():
        n = 5
        env = _FakeEnv(n).observe_as(layer)
        assert layer.observe(env).shape == (n, GAIN_LAYERS[layer.name].obs_dim)
        assert output_dim(layer) == (LAYERS[layer.below].command_dim if layer.below else 4)
    assert [layer.name for layer in layers_below("position")] == ["velocity", "attitude", "rate"]


def test_attitude_layer_frames_with_a_tilted_drone():
    layer, n = LAYERS["attitude"], 6
    gains = GAIN_LAYERS["attitude"]
    env = _FakeEnv(n).observe_as(layer, command=torch.randn(n, 4), history=torch.zeros(n, gains.history, gains.action_dim))
    command = env.cascade.observing.command
    obs = layer.observe(env)
    rotation = matrix_from_quat(env.values["root_quat_w"])       # the check is done with the matrix, independent of quat_apply
    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = -torch.asin(rotation[:, 2, 0])
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    expected = wrap_to_pi(command[:, :3] - torch.stack([roll, pitch, yaw], dim=-1))
    assert torch.allclose(obs[:, :3], expected, atol=1e-5)
    assert torch.allclose(obs[:, 3], command[:, 3] / WEIGHT_N, atol=1e-6)
    gravity_in_body = (rotation.transpose(1, 2) @ torch.tensor([0.0, 0.0, -1.0])).squeeze(-1)
    assert torch.allclose(obs[:, 4:7], gravity_in_body, atol=1e-5)


def test_cascade_runs_frozen_layers_top_down_at_their_rates():
    calls = []

    class Fake:
        def __init__(self, name, period, width):
            self.name, self.period, self.output = name, period, torch.zeros(1, width)

        def step(self, env, command):
            calls.append((action._tick, self.name, command.clone(), action.observing is self))
            self.output = torch.full_like(self.output, float(len(calls)))

    action = GainCascadeAction.__new__(GainCascadeAction)
    action.below = [Fake("velocity", 10, 4), Fake("attitude", 10, 4), Fake("rate", 5, 4)]
    action.observing = action
    action._output = torch.full((1, 4), -1.0)
    action._motor = torch.zeros(1, 4)
    action._propulsion = SimpleNamespace(step=lambda *args: None)
    action.cfg = SimpleNamespace(pwm_max=1.0)
    action._robot, action._body_id, action._physics_dt, action._env = None, 0, 1.0 / FW_TICK_HZ, None
    action._tick = 0
    for _ in range(10):
        action.apply_actions()

    assert [(tick, name) for tick, name, _, _ in calls] == [(0, "velocity"), (0, "attitude"), (0, "rate"), (5, "rate")]
    assert all(observed for _, _, _, observed in calls)             # each layer is the one observed while it runs
    assert action.observing is action                               # and the layer in training afterwards
    assert torch.equal(calls[0][2], torch.full((1, 4), -1.0))       # velocity gets the trained layer's output
    assert torch.equal(calls[1][2], torch.full((1, 4), 1.0))        # attitude gets the velocity output
    assert torch.equal(calls[3][2], torch.full((1, 4), 2.0))        # rate at tick 5: attitude output held
    assert torch.equal(action._motor, torch.full((1, 4), 4.0))      # motors: the last rate output


def test_env_cfg_lists_the_observation_terms_of_every_layer():
    """The env cfg of a layer lists, in order, the terms the frozen layer computes (``Layer.observation``)."""
    from Drone_RL.uav.rl_control.attitude_env_cfg import AttitudeEnvCfg
    from Drone_RL.uav.rl_control.position_env_cfg import PositionEnvCfg
    from Drone_RL.uav.rl_control.rate_env_cfg import RateEnvCfg
    from Drone_RL.uav.rl_control.velocity_env_cfg import VelocityEnvCfg

    for env_cfg in (RateEnvCfg, AttitudeEnvCfg, VelocityEnvCfg, PositionEnvCfg):
        cfg = env_cfg()
        layer = LAYERS[cfg.LAYER]
        terms = [t for t in vars(cfg.observations.policy).values() if isinstance(t, ObservationTermCfg)]
        assert [t.func for t in terms] == list(layer.observation)
        assert abs(cfg.sim.dt * cfg.decimation - 1.0 / layer.hz) < 1e-9       # one env step = one period of the layer
        env = _FakeEnv(4).observe_as(layer)
        assert torch.cat([t.func(env) for t in terms], dim=-1).shape[-1] == GAIN_LAYERS[layer.name].obs_dim


def test_path_slope_is_the_derivative_of_the_path():
    from Drone_RL.uav.mdp.commands import PATH_TYPES, path_point

    n = 64
    path_type = torch.arange(n) % len(PATH_TYPES)
    radius, spin, phase = torch.rand(n) + 0.3, torch.where(torch.rand(n) < 0.5, 1.0, -1.0), torch.rand(n) * 6.28
    position, slope = path_point(path_type, radius, spin, phase)
    eps = 1e-3
    ahead, _ = path_point(path_type, radius, spin, phase + eps)
    assert torch.allclose((ahead - position) / eps, slope, atol=2e-2)
    start, _ = path_point(path_type, radius, spin, torch.zeros(n))
    assert torch.allclose(start[path_type == 0, 0], radius[path_type == 0])       # circle starts on +x
    assert torch.allclose(start[path_type == 1], torch.zeros(int((path_type == 1).sum()), 2), atol=1e-6)   # figure 8 at the center


def test_position_command_puts_the_path_under_the_drone_and_carries_the_target_velocity():
    from Drone_RL.uav.mdp.commands import LayerCommand
    from Drone_RL.uav.rl_control.position_env_cfg import PositionEnvCfg

    n = 32
    cfg = PositionEnvCfg().commands.layer
    env = _FakeEnv(n)
    env.values["root_pos_w"][:, 2] = 1.5
    command = object.__new__(LayerCommand)           # only the path functions run: no simulator, no command manager
    command.__dict__.update(cfg=cfg, _env=env, device="cpu", num_envs=n, target=torch.zeros(n, 4),
                            target_velocity=torch.zeros(n, 3), _goal=torch.zeros(n, 3),
                            _on_path=torch.zeros(n, dtype=torch.bool), _center=torch.zeros(n, 2),
                            _height=torch.zeros(n), _radius=torch.ones(n), _speed=torch.zeros(n), _spin=torch.ones(n),
                            _phase=torch.zeros(n), _path_type=torch.zeros(n, dtype=torch.long),
                            _path_codes=torch.arange(2))
    command._resample_command(list(range(n)))
    on = command._on_path
    assert on.any() and not on.all()
    start = env.values["root_pos_w"]
    assert torch.allclose(command.target[on, :3], start[on], atol=1e-5)               # the path starts under the drone
    assert torch.allclose(command.target[~on, :3], command._goal[~on])                # fixed target elsewhere
    assert (command.target_velocity[~on] == 0).all()
    before = command.target[:, :3].clone()
    command._follow_path(0.02)
    speed = (command.target[:, :3] - before).norm(dim=-1)[on] / 0.02
    assert torch.allclose(speed, command.target_velocity.norm(dim=-1)[on], rtol=0.1, atol=0.03)
