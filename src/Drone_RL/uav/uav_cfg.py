"""Configuration for the Crazyflie 2.1 Brushless (``assets/data/crazyflie/cf2x.usd``).

The USD is the Crazyflie 2.x model of Isaac Sim; the brushless variant has the same layout, so only the
numbers change (the collision and visual geometry stay the 2.x one). 
Articulation root ``/crazyflie``, rigid body ``body``, four propeller bodies ``m1_prop`` .. ``m4_prop`` 
on revolute joints ``m1_joint`` .. ``m4_joint`` (axis z). 
Frame: x forward, y left, z up.
"""

import math

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from Drone_RL.assets import DRONE_RL_ASSETS_DIR

_DRONE_USD_PATH = str(DRONE_RL_ASSETS_DIR / "crazyflie" / "cf2x.usd")

# ── Geometry ────────────────────────────────────────────────────────────────────────────
"""Motor offset on each axis [m]: the frame is 100 mm motor to motor on the diagonal,
so 35.36 mm, the L is the distance from the center of the drone to the center of each motor (bitcraze_crazyflie_2.1_brushless_datasheet_rev3.pdf)"""
CF_MOTOR_HALF_XY_M = 0.050 / math.sqrt(2.0)

"""Motor centres (x, y) [m] in the body frame, order m1..m4 of the joints in cf2x.usd."""
CF_MOTOR_XY = (
    (CF_MOTOR_HALF_XY_M, -CF_MOTOR_HALF_XY_M),
    (-CF_MOTOR_HALF_XY_M, -CF_MOTOR_HALF_XY_M),
    (-CF_MOTOR_HALF_XY_M, CF_MOTOR_HALF_XY_M),
    (CF_MOTOR_HALF_XY_M, CF_MOTOR_HALF_XY_M),
)

"""+1 = clockwise propeller seen from above (reaction torque +z on the body), per motor m1..m4."""
CF_MOTOR_SPIN = (-1.0, 1.0, -1.0, 1.0)

# ── Mass ────────────────────────────────────────────────────────────────────────────────
"""All-up mass [kg]: the 45 g configuration of Busetto et al. (section 4.1)."""
DRONE_MASS_TOTAL_KG = 0.045

"""Mass of each propeller body [kg]. Not published for the brushless propellers: the cf2x.usd value."""
DRONE_MASS_FAN_KG = 0.0008

"""Mass of the ``body`` rigid body [kg] (name kept from the previous airframe)."""
DRONE_MASS_GROUP1_KG = DRONE_MASS_TOTAL_KG - 4 * DRONE_MASS_FAN_KG

"""Principal moments [kg m^2]: Busetto et al. (section 4.1), measured on the 45 g configuration."""
DRONE_INERTIA_DIAG = (2.3951e-5, 2.3951e-5, 3.2347e-5)

"""Weight of the nominal configuration [N]."""
DRONE_HOVER_THRUST_N = DRONE_MASS_TOTAL_KG * 9.81

# ── Motor and propulsion ────────────────────────────────────────────────────────────────
# Thrust-stand identification, Folk, arXiv:2604.00343, section 6.3.1, fig. 6.5:
#   thrust of one motor  T = CF_K_ETA * eta^2
#   rotor speed          eta = CF_KV * (V + CF_V0) * (PWM - CF_DZ)^(2/3)
# Measured for PWM 10000..45000 and V = 2.7..4.1 V; above 45000 it is extrapolated.
"""Thrust coefficient [N/(rad/s)^2]."""
CF_K_ETA = 4.052e-8

"""Motor constant [(rad/s)/V]."""
CF_KV = 0.3637

"""Voltage offset [V]."""
CF_V0 = 0.5535

"""PWM dead zone."""
CF_DZ = 4673.0

"""Full-scale motor command (16-bit)."""
CF_PWM_MAX = 65535.0

"""Battery voltage of the simulation [V]."""
CF_BATTERY_V_NOM = 3.7

"""Yaw torque per unit of thrust [m], 2.08e-3: kM / kF of Busetto et al., arXiv:2512.14450, eq. 5 (kM = 7.73e-11,
kF = 3.72e-8). Only this ratio is used; the thrust itself comes from the Folk curve above."""
DRONE_KM = 7.73e-11 / 3.72e-8

"""Induced-drag coefficient [N s^2/(m rad)] (Folk, table 6.2)."""
CF_KD_DRAG = 5.09e-6


"""Scale on the thrust of the Folk curve: 2 doubles the thrust at every PWM (thrust to weight 2.1 -> 4.2). At 1 the
hover throttle was 0.61 and little was left for the torques; the rate policy was weak and the drone jerked around
the target. The yaw torque (DRONE_KM times the thrust) scales with it."""
CF_THRUST_SCALE = 2.0


def motor_thrust_n(pwm, voltage: float = CF_BATTERY_V_NOM):
    """Thrust of one motor [N] from the PWM command and the battery voltage (numpy or torch)."""
    speed = CF_KV * (voltage + CF_V0) * (pwm - CF_DZ).clip(0.0) ** (2.0 / 3.0)
    return CF_THRUST_SCALE * CF_K_ETA * speed**2


"""Maximum thrust per motor [N] at full command and nominal voltage (0.232 N extrapolated from Folk, times
CF_THRUST_SCALE). The datasheet quotes up to 30 g (0.294 N) with a full battery."""
CF_F_MAX_N = float(motor_thrust_n(np.array(CF_PWM_MAX)))



def _fit_thrust_curve():
    """Quadratic fit of the motor model at the nominal voltage over PWM CF_DZ..CF_PWM_MAX.

    Grams for the SUM of the four motors, so the action terms divide by 4. Error at most 1.9 g on that
    sum, 0.6 g inside the measured PWM range."""
    pwm = np.linspace(CF_DZ, CF_PWM_MAX, 400)
    grams = 4.0 * motor_thrust_n(pwm) * 1000.0 / 9.80665
    qa, qb, qc = np.polyfit(pwm, grams, 2)
    return float(qa), float(qb), float(qc)

"""Total thrust of the four motors in grams, a*pwm^2 + b*pwm + c, all motors at the same command."""
CF_THRUST_COEF_G = _fit_thrust_curve()


def _hover_throttle() -> float:
    a, b, c = CF_THRUST_COEF_G
    pwm = (-b + math.sqrt(b * b - 4.0 * a * (c - DRONE_MASS_TOTAL_KG * 1000.0))) / (2.0 * a)
    return pwm / CF_PWM_MAX

"""Hover throttle at the nominal voltage (0.50 for 34 g)."""
DRONE_HOVER_THROTTLE = _hover_throttle()

"""Linear air drag [N s/m] at hover (0.024 with CF_THRUST_SCALE 2): kd times the sum of the four rotor speeds at hover.
The horizontal drag force is -CF_DRAG_COEF * v."""
CF_DRAG_COEF = CF_KD_DRAG * 4 * math.sqrt(DRONE_MASS_TOTAL_KG * 9.81 / 4 / (CF_THRUST_SCALE * CF_K_ETA))


# ── Articulation ────────────────────────────────────────────────────────────────────────
"""Revolute joints of the propellers, order m1..m4. Visual only: the thrust is applied to the body as a wrench."""
CF_MOTOR_JOINTS = ("m1_joint", "m2_joint", "m3_joint", "m4_joint")


def apply_real_mass_inertia(root_prim_path: str) -> None:
    """Author the nominal masses and the body inertia on the spawned USD.

    Call after the USD is spawned and before ``sim.reset()``: PhysX reads the mass API only when
    it cooks the articulation. Only the rigid body named ``body`` changes; the propellers keep the USD mass.
    """
    import omni.usd
    from pxr import Gf, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"apply_real_mass_inertia: prim does not exist: {root_prim_path}")

    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if name == "body":
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(DRONE_MASS_GROUP1_KG)
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*DRONE_INERTIA_DIAG))


@sim_utils.clone
def _spawn_uav_with_nominal_mass(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the USD, then author the nominal mass and inertia before the physics cook."""
    prim = sim_utils.spawn_from_usd.__wrapped__(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    apply_real_mass_inertia(str(prim.GetPath()))
    return prim


UAV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_uav_with_nominal_mass,
        usd_path=_DRONE_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # The contact sensor of the RL scene only reads forces from prims that have the contact
        # report schema, which this flag applies.
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
