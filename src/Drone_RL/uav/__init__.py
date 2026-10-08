"""UAV: Crazyflie 2.1 Brushless tasks (``Isaac-UAV-*``).

Tasks (all in ``rl_control/``): ``Isaac-UAV-{Rate,Attitude,Velocity,Position}-Gains-v0`` (the gain cascade, one layer
per task) and ``Isaac-UAV-Landing-ArUco-v0``. Their MDP terms are shared in ``mdp/``. Physical constants live in ``uav_cfg.py``; the USD is
``assets/data/crazyflie/cf2x.usd``.
"""

from . import rl_control  # noqa: F401
from .uav_cfg import *
