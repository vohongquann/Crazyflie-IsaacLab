"""Motor geometry of ``cf2x.usd`` against ``uav_cfg``. Run with ``pytest tests/test_usd_geometry.py -s`` to see the table.

The USD is the Crazyflie 2.x (31 mm arms); ``uav_cfg`` keeps its motor order and signs but uses the Brushless arm
(100 mm motor to motor on the diagonal, 35.36 mm per axis). The spin signs are not in the USD (the joints only give the
axis), so they are printed for reference and not checked here.
"""

import math

import pytest

from Drone_RL.uav import uav_cfg as U

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402


def _motor_joints() -> list[tuple[str, tuple, str]]:
    """(path, localPos0, axis) of the revolute joints m1..m4, in the order of their names."""
    stage = Usd.Stage.Open(U._DRONE_USD_PATH)
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            joints.append((str(prim.GetPath()), tuple(joint.GetLocalPos0Attr().Get()), joint.GetAxisAttr().Get()))
    return sorted(joints, key=lambda j: j[0])


def test_print_motor_geometry():
    joints = _motor_joints()
    print(f"\nUSD: {U._DRONE_USD_PATH}")
    print(f"{'joint':28} {'USD x, y, z [mm]':24} {'axis':5} {'arm [mm]':9} {'cfg x, y [mm]':18} spin")
    for (path, pos, axis), xy, spin in zip(joints, U.CF_MOTOR_XY, U.CF_MOTOR_SPIN):
        usd = ", ".join(f"{1000 * v:+.1f}" for v in pos)
        cfg = ", ".join(f"{1000 * v:+.2f}" for v in xy)
        arm = 1000 * math.hypot(pos[0], pos[1])
        print(f"{path:28} {usd:24} {axis:5} {arm:9.1f} {cfg:18} {'CW' if spin > 0 else 'CCW'}")
    print(f"cfg L = {1000 * U.CF_MOTOR_HALF_XY_M:.2f} mm per axis, "
          f"{1000 * math.hypot(U.CF_MOTOR_HALF_XY_M, U.CF_MOTOR_HALF_XY_M):.1f} mm centre to motor")
    assert len(joints) == 4


def test_cfg_motor_order_matches_the_usd():
    """Same quadrant per motor: the policy's action i drives joint m(i+1)."""
    joints = _motor_joints()
    assert [path.rsplit("/", 1)[-1] for path, _, _ in joints] == ["m1_joint", "m2_joint", "m3_joint", "m4_joint"]
    for (_, pos, axis), xy in zip(joints, U.CF_MOTOR_XY):
        assert axis == "Z"
        assert math.copysign(1.0, pos[0]) == math.copysign(1.0, xy[0])
        assert math.copysign(1.0, pos[1]) == math.copysign(1.0, xy[1])


def test_arm_length_is_the_brushless_one():
    """100 mm motor to motor on the diagonal -> 50 mm centre to motor -> 35.36 mm per axis."""
    assert U.CF_MOTOR_HALF_XY_M == pytest.approx(0.050 / math.sqrt(2.0))
    for x, y in U.CF_MOTOR_XY:
        assert abs(x) == abs(y) == U.CF_MOTOR_HALF_XY_M


if __name__ == "__main__":
    test_print_motor_geometry()
    test_cfg_motor_order_matches_the_usd()
    test_arm_length_is_the_brushless_one()
    print("all checks passed")
