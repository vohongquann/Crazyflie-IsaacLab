r"""RL tasks of the Crazyflie: one file per task, built from the shared terms of ``uav/mdp/``.

RL cascade, one network per PID layer, trained bottom-up, each on top of the frozen layers below it:

    Isaac-UAV-Rate-RL-v0       rate_env_cfg.py       body rates, thrust   -> motors          (500 Hz)
    Isaac-UAV-Attitude-RL-v0   attitude_env_cfg.py   roll, pitch, yaw, thrust -> rates (+ thrust through)  (250 Hz, frozen rate below)
    Isaac-UAV-Velocity-RL-v0   velocity_env_cfg.py   velocity             -> roll, pitch, thrust   (100 Hz, frozen attitude, rate)
    Isaac-UAV-Position-RL-v0   position_env_cfg.py   position             -> velocity        (50 Hz, frozen velocity, ...)

    cascade_env_cfg.py          scene, command, action, observation, events, terminations shared by the four
    frozen/rl/                  <layer>.pt of a trained layer (the ``exported/policy.pt`` of ``play``), which the layers above load

Gain tasks: every layer is a network that writes the 9 PID gains of its layer (kp, ki, kd x 3 axes; the zero action is the
tuned PID), trained bottom-up like the cascade above, each over the frozen gain networks of the layers below
(``gains_env_cfg.py``, ``mdp/gains.py``). Their frozen files are in their own folder, ``frozen/gains/<layer>.pt``, next to
``frozen/rl/<layer>.pt`` of the cascade above:

    Isaac-UAV-Rate-Gains-v0 -> Attitude-Gains -> Velocity-Gains -> Position-Gains   (log folders uav_<layer>_gains)

Landing on an ArUco marker (end-to-end PPO, four motor commands, downward camera, rendered automatically):

    Isaac-UAV-Landing-ArUco-v0 landing_env_cfg.py    (+ marker_plate.py, the pad in the scene)
    Isaac-UAV-Landing-ArUco-Cascade-v0  same, the policy writes a wanted velocity to the frozen RL cascade (needs frozen/rl/)

PPO settings: ``agents/<task>_ppo_cfg.py``. Landing: guide/06_aruco_landing.md.

Training commands, order and the frozen files of both cascades: guide/08_training.md.

"""
import gymnasium as gym

_TASKS = {
    "Isaac-UAV-Rate-RL-v0": ("rate_env_cfg:RateEnvCfg", "rate_ppo_cfg:RatePPORunnerCfg"),
    "Isaac-UAV-Attitude-RL-v0": ("attitude_env_cfg:AttitudeEnvCfg", "attitude_ppo_cfg:AttitudePPORunnerCfg"),
    "Isaac-UAV-Velocity-RL-v0": ("velocity_env_cfg:VelocityEnvCfg", "velocity_ppo_cfg:VelocityPPORunnerCfg"),
    "Isaac-UAV-Position-RL-v0": ("position_env_cfg:PositionEnvCfg", "position_ppo_cfg:PositionPPORunnerCfg"),
    "Isaac-UAV-Rate-Gains-v0": ("gains_env_cfg:RateGainsEnvCfg", "gains_ppo_cfg:RateGainsPPORunnerCfg"),
    "Isaac-UAV-Attitude-Gains-v0": ("gains_env_cfg:AttitudeGainsEnvCfg", "gains_ppo_cfg:AttitudeGainsPPORunnerCfg"),
    "Isaac-UAV-Velocity-Gains-v0": ("gains_env_cfg:VelocityGainsEnvCfg", "gains_ppo_cfg:VelocityGainsPPORunnerCfg"),
    "Isaac-UAV-Position-Gains-v0": ("gains_env_cfg:PositionGainsEnvCfg", "gains_ppo_cfg:PositionGainsPPORunnerCfg"),
    "Isaac-UAV-Landing-ArUco-v0": ("landing_env_cfg:LandingEnvCfg", "landing_ppo_cfg:LandingPPORunnerCfg"),
    "Isaac-UAV-Landing-ArUco-Cascade-v0": ("landing_env_cfg:LandingCascadeEnvCfg",
                                           "landing_ppo_cfg:LandingCascadePPORunnerCfg"),
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

