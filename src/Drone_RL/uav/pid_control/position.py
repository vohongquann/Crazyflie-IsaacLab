"""Position layer: position error -> wanted velocity.

Test: all four layers fly (the whole cascade). The drone takes off from the ground and follows a 0.5 m square.

    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/position.py        # Isaac Sim window + live plot
    ~/miniconda3/envs/env_isaaclab/bin/python src/Drone_RL/uav/pid_control/position.py --viz none --no_live   # PNG only
"""
import torch
from Drone_RL.uav.pid_control.flight_test import LayerTest, run, step_table
from Drone_RL.uav.pid_control.pid import PID


POSITION_KP = (3.9, 3.9, 6.8)                # (x, y, z), output [m/s]
POSITION_MAX_VELOCITY = (1.6, 1.6, 1.6)      # [m/s] limit of the output


class PositionController:
    def __init__(self, device):
        kp = torch.tensor(POSITION_KP, device=device)
        out_limit = torch.tensor(POSITION_MAX_VELOCITY, device=device)
        self.pid = PID(kp=kp, out_limit=out_limit)

    def reset(self, env_ids=None) -> None:
        self.pid.reset(env_ids)

    def update(self, target: torch.Tensor, pos: torch.Tensor, dt: float) -> torch.Tensor:
        """Target position and position (world, (N, 3) [m]) -> wanted velocity [m/s]."""
        return self.pid.update(target - pos, dt)


# (t_start [s], x, y, z) in m
STEPS = [
    (0.0, 0.0, 0.0, 0.5),
    (3.0, 0.5, 0.0, 0.5), (6.0, 0.5, 0.5, 0.5), (9.0, 0.0, 0.5, 0.5), (12.0, 0.0, 0.0, 0.5),
]


def _control(controller, wanted, state, dt):
    target = torch.tensor([wanted], dtype=torch.float32, device=state.pos.device)
    return controller.step(target, state.pos, state.vel, state.quat, state.rates, dt)


TEST = LayerTest(
    name="position", title="Position layer: position -> velocity (whole cascade)",
    labels=["X [m]", "Y [m]", "Z [m]"],
    start_z=0.1, duration=15.0,
    setpoint=lambda t: [float(v) for v in step_table(STEPS, t)],
    control=_control,
    measure=lambda state: state.pos[0].tolist(),
)

if __name__ == "__main__":
    run(TEST)
