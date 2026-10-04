"""Physics and loop rates shared by the action terms."""
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONTROL LOOP RATE of the Crazyflie firmware (src/modules/interface/stabilizer_types.h):
#   RATE_MAIN_LOOP 1000 Hz (sensor sampling), ATTITUDE_RATE 500 Hz (attitude + rate PID, motor
#   output), POSITION_RATE 100 Hz (position / velocity controller).
# The physics runs at RATE_MAIN_LOOP; the name keeps the FW_ prefix of the previous firmware.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

"""RATE_MAIN_LOOP: the fastest loop of the firmware, so the physics step is 1/it (1 ms)."""
FW_TICK_HZ = 1000.0

"""Loop rates of this project [Hz]: the RL cascade layers (``<Layer>.hz`` in mdp/layers.py) and the landing policy
(``decimation`` of landing_env_cfg.py). Each must divide ``FW_TICK_HZ``: a loop runs every FW_TICK_HZ / rate physics steps."""
RATE_HZ = 500.0
ATTITUDE_HZ = 250.0
VELOCITY_HZ = 100.0
POSITION_HZ = 50.0
LANDING_HZ = 25.0
