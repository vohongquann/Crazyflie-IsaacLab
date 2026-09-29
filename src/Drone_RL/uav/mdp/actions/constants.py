"""Physics rate and motor lag constants shared by the action terms."""
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CONTROL LOOP RATE of the Crazyflie firmware (src/modules/interface/stabilizer_types.h):
#   RATE_MAIN_LOOP 1000 Hz (sensor sampling), ATTITUDE_RATE 500 Hz (attitude + rate PID, motor
#   output), POSITION_RATE 100 Hz (position / velocity controller).
# The physics runs at ATTITUDE_RATE; the name keeps the FW_ prefix of the previous firmware.
# ═══════════════════════════════════════════════════════════════════════════════════════════════

FW_TICK_HZ = 500.0
"""ATTITUDE_RATE: rate loop and motor output. The fastest loop simulated, so sim dt should be 1/it."""

MOTOR_TAU_INC_RANGE = (0.05, 0.08)
MOTOR_TAU_DEC_RANGE = (0.005, 0.005)
"""Motor lag time constants [s], drawn per motor and per episode: ``INC`` while the motor speeds up, ``DEC`` while it
slows down. Values of the ARL robot in Isaac Lab (``ARL_ROBOT_1_THRUSTER``: ``tau_inc_range``, ``tau_dec_range``).
No value is published for the brushless Crazyflie."""
