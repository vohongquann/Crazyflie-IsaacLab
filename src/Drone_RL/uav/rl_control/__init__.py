r"""RL tasks of the Crazyflie: one file per task, built from the shared terms of ``uav/mdp/``.

RL cascade, one network per PID layer, trained bottom-up, each on top of the frozen layers below it:

    Isaac-UAV-Rate-RL-v0       rate_env_cfg.py       body rates, thrust   -> motors          (500 Hz)
    Isaac-UAV-Attitude-RL-v0   attitude_env_cfg.py   acceleration, yaw    -> rates, thrust   (250 Hz, frozen rate below)
    Isaac-UAV-Attitude-PIDRate-RL-v0  attitude_env_cfg.py  same, with the PID rate controller below instead
    Isaac-UAV-Velocity-RL-v0   velocity_env_cfg.py   velocity             -> acceleration    (100 Hz, frozen attitude, rate)
    Isaac-UAV-Position-RL-v0   position_env_cfg.py   position             -> velocity        (50 Hz, frozen velocity, ...)

    cascade_env_cfg.py          scene, command, action, observation, events, terminations shared by the four
    frozen/                     <layer>.pt of a trained layer (the ``exported/policy.pt`` of ``play``), which the layers above load

Landing on an ArUco marker (end-to-end PPO, four motor commands, downward camera, rendered automatically):

    Isaac-UAV-Landing-ArUco-v0 landing_env_cfg.py    (+ marker_plate.py, the pad in the scene)

PPO settings: ``agents/<task>_ppo_cfg.py``. Workflow: guide/05_rl_cascade.md, guide/06_aruco_landing.md.

Commands, from the repo root. The cascade is trained bottom-up; each layer is trained on the frozen ones below it.
``--viz kit`` opens the Isaac Sim window (slower, so fewer envs, e.g. ``--num_envs 64``); without it: headless.
Close the ``play`` window before the ``cp``.

    mkdir -p src/Drone_RL/uav/rl_control/frozen          # once

    # 1. rate (no frozen layer)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Rate-RL-v0 --num_envs 1024 \
        --video --video_length 400 --video_interval 5000                  # [--viz kit]
    isaaclab play  --rl_library rsl_rl --task Isaac-UAV-Rate-RL-v0 --num_envs 16 --viz kit
    cp "$(ls -d logs/rsl_rl/uav_rate/*/ | tail -1)exported/policy.pt" src/Drone_RL/uav/rl_control/frozen/rate.pt

    # 2. attitude (needs frozen/rate.pt)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Attitude-RL-v0 --num_envs 1024 \
        --video --video_length 400 --video_interval 5000                  # [--viz kit]
    isaaclab play  --rl_library rsl_rl --task Isaac-UAV-Attitude-RL-v0 --num_envs 16 --viz kit
    cp "$(ls -d logs/rsl_rl/uav_attitude/*/ | tail -1)exported/policy.pt" src/Drone_RL/uav/rl_control/frozen/attitude.pt

    # 2b. attitude on the PID rate controller (no frozen/rate.pt; own log folder uav_attitude_pidrate)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Attitude-PIDRate-RL-v0 --num_envs 1024 \
        --video --video_length 400 --video_interval 5000                  # [--viz kit]

    # 3. velocity (needs frozen/rate.pt, attitude.pt)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Velocity-RL-v0 --num_envs 1024 \
        --video --video_length 400 --video_interval 5000                  # [--viz kit]
    isaaclab play  --rl_library rsl_rl --task Isaac-UAV-Velocity-RL-v0 --num_envs 16 --viz kit
    cp "$(ls -d logs/rsl_rl/uav_velocity/*/ | tail -1)exported/policy.pt" src/Drone_RL/uav/rl_control/frozen/velocity.pt

    # 4. position (needs frozen/rate.pt, attitude.pt, velocity.pt)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Position-RL-v0 --num_envs 1024 \
        --video --video_length 400 --video_interval 5000                  # [--viz kit]
    isaaclab play  --rl_library rsl_rl --task Isaac-UAV-Position-RL-v0 --num_envs 16 --viz kit
    cp "$(ls -d logs/rsl_rl/uav_position/*/ | tail -1)exported/policy.pt" src/Drone_RL/uav/rl_control/frozen/position.pt

    # landing on the ArUco marker (separate task, no frozen layers)
    isaaclab train --rl_library rsl_rl --task Isaac-UAV-Landing-ArUco-v0 --num_envs 32 \
        --video --video_length 400 --video_interval 5000

Quick check that an env builds (a few seconds): add ``--num_envs 64 --max_iterations 2`` to a train command.
"""
import gymnasium as gym

_TASKS = {
    "Isaac-UAV-Rate-RL-v0": ("rate_env_cfg:RateEnvCfg", "rate_ppo_cfg:RatePPORunnerCfg"),
    "Isaac-UAV-Attitude-RL-v0": ("attitude_env_cfg:AttitudeEnvCfg", "attitude_ppo_cfg:AttitudePPORunnerCfg"),
    "Isaac-UAV-Attitude-PIDRate-RL-v0": ("attitude_env_cfg:AttitudePIDRateEnvCfg",
                                         "attitude_ppo_cfg:AttitudePIDRatePPORunnerCfg"),
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

