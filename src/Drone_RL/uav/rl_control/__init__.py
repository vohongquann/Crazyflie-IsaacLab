r"""RL tasks of the Crazyflie: one file per task, built from the shared terms of ``uav/mdp/``.

Gain cascade: every layer is a network that writes the 9 PID gains of its layer (kp, ki, kd x 3 axes; the zero action
is the tuned PID), trained bottom-up, each on top of the frozen gain networks of the layers below:

    Isaac-UAV-Rate-Gains-v0       body rates, thrust       -> motors              (500 Hz)
    Isaac-UAV-Attitude-Gains-v0   roll, pitch, yaw, thrust -> rates (+ thrust)    (250 Hz, frozen rate below)
    Isaac-UAV-Velocity-Gains-v0   velocity                 -> roll, pitch, thrust (100 Hz, frozen attitude, rate)
    Isaac-UAV-Position-Gains-v0   position                 -> velocity            (50 Hz, frozen velocity, ...)

    <layer>_env_cfg.py          command, observation, rewards, resets of each layer
    cascade_env_cfg.py          scene, action, events, terminations shared by the four
    gains_env_cfg.py            the four tasks (``mdp/gains.py``, ``mdp/actions/gain_action.py``)
    frozen/gains/               <layer>.pt of a trained layer (the ``exported/policy.pt`` of ``play``), which the layers
                                above load; log folders uav_<layer>_gains

Landing on an ArUco marker (end-to-end PPO, four motor commands, downward camera, rendered automatically):

    Isaac-UAV-Landing-ArUco-v0 landing_env_cfg.py    (+ marker_plate.py, the pad in the scene)

PPO settings: ``agents/<task>_ppo_cfg.py``. Landing: guide/06_aruco_landing.md.

"""
import gymnasium as gym

_TASKS = {
    "Isaac-UAV-Rate-Gains-v0": ("gains_env_cfg:RateGainsEnvCfg", "gains_ppo_cfg:RateGainsPPORunnerCfg"),
    "Isaac-UAV-Attitude-Gains-v0": ("gains_env_cfg:AttitudeGainsEnvCfg", "gains_ppo_cfg:AttitudeGainsPPORunnerCfg"),
    "Isaac-UAV-Velocity-Gains-v0": ("gains_env_cfg:VelocityGainsEnvCfg", "gains_ppo_cfg:VelocityGainsPPORunnerCfg"),
    "Isaac-UAV-Position-Gains-v0": ("gains_env_cfg:PositionGainsEnvCfg", "gains_ppo_cfg:PositionGainsPPORunnerCfg"),
    "Isaac-UAV-Landing-ArUco-v0": ("landing_env_cfg:LandingEnvCfg", "landing_ppo_cfg:LandingPPORunnerCfg"),
}

for _task, (_env, _agent) in _TASKS.items():
    gym.register(
        id=_task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.{_agent}",
            "default_agent": "rsl_rl",
        },
    )

