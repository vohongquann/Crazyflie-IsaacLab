"""MDP terms shared by every UAV task (actions, commands, observations, rewards, terminations).

    actions/        MotorAction (policy -> 4 motors), CascadeAction (RL layer -> frozen layers -> motors), Propulsion
    layers.py       the four RL cascade layers: what each observes and outputs
    commands.py     LayerCommand: command of the RL layer in training (PID layers above + random target)
    aruco.py        ArUco marker texture and OpenCV detector (landing)
    observations.py state observation (Eschmann et al. 2024), RL layer observation, ArUco observation
    rewards.py      tracking rewards of the RL layers, landing rewards
    terminations.py flight envelope (RL layers), landed / crashed (landing)

The task files in ``rl_control/`` only assemble these terms into environments.
"""
from .actions import *
from .commands import *
from .observations import *
from .rewards import *
from .terminations import *
