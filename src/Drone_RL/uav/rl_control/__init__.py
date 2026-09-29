"""RL tasks of the Crazyflie: one file per task, built from the shared terms of ``uav/mdp/``.

RL cascade, one network per PID layer, trained bottom-up, each on top of the frozen layers below it:

    Isaac-UAV-Rate-RL-v0       rate_env_cfg.py       body rates, thrust   -> motors          (100 Hz)
    Isaac-UAV-Attitude-RL-v0   attitude_env_cfg.py   acceleration, yaw    -> rates, thrust   (50 Hz, frozen rate below)
    Isaac-UAV-Velocity-RL-v0   velocity_env_cfg.py   velocity             -> acceleration    (50 Hz, frozen attitude, rate)
    Isaac-UAV-Position-RL-v0   position_env_cfg.py   position             -> velocity        (50 Hz, frozen velocity, ...)

    cascade_env_cfg.py          scene, command, action, observation, events, terminations shared by the four
    freeze.py, frozen/          freeze a trained layer into frozen/<layer>.pt, which the layers above load

Landing on an ArUco marker (end-to-end PPO, four motor commands, downward camera, rendered automatically):

    Isaac-UAV-Landing-ArUco-v0 landing_env_cfg.py    (+ marker_plate.py, the pad in the scene)

PPO settings: ``agents/<task>_ppo_cfg.py``. Workflow: guide/05_rl_cascade.md, guide/06_aruco_landing.md.
"""
import gymnasium as gym

_TASKS = {
    "Isaac-UAV-Rate-RL-v0": ("rate_env_cfg:RateEnvCfg", "rate_ppo_cfg:RatePPORunnerCfg"),
    "Isaac-UAV-Attitude-RL-v0": ("attitude_env_cfg:AttitudeEnvCfg", "attitude_ppo_cfg:AttitudePPORunnerCfg"),
    "Isaac-UAV-Velocity-RL-v0": ("velocity_env_cfg:VelocityEnvCfg", "velocity_ppo_cfg:VelocityPPORunnerCfg"),
    "Isaac-UAV-Position-RL-v0": ("position_env_cfg:PositionEnvCfg", "position_ppo_cfg:PositionPPORunnerCfg"),
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
