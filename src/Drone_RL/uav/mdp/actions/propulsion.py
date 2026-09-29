"""Propulsion of the Crazyflie: PWM of the four motors -> force and torque on the body.

Every physics step (formulas in guide/02_propulsion.md, section 2.3):

    PWM  --pwm_to_thrust-->  F_cmd  --motor_lag_step-->  F  --(noise)-->  F  --wrench-->  force, torque on the body

The controller (or the policy) only writes a new PWM at its own rate and the value is held in between, while the
motors and the frame keep reacting, as on the real drone.
"""
import torch

from Drone_RL.uav.uav_cfg import CF_K_ETA, CF_MOTOR_JOINTS


def pwm_to_thrust(pwm: torch.Tensor, a: float, b: float, c: float, g: float, f_max: float) -> torch.Tensor:
    """PWM (N, 4) -> thrust of each motor (N, 4) [N].

    ``a*pwm^2 + b*pwm + c`` (``uav_cfg.CF_THRUST_COEF_G``) is the thrust of the four motors together, in grams.
    """
    grams_all = a * pwm * pwm + b * pwm + c        # total thrust [g], all four motors at this PWM
    newton_one = grams_all / 4.0 * g / 1000.0      # one motor: / 4, then grams -> kg (/ 1000), then kg -> N (* g)
    return newton_one.clamp(0.0, f_max)            # no negative thrust below the dead zone, cap at full command


def thrust_to_pwm(force: torch.Tensor, a: float, b: float, c: float, g: float, pwm_max: float) -> torch.Tensor:
    """Inverse of ``pwm_to_thrust``: thrust of each motor (N, 4) [N] -> PWM (N, 4). Solves ``a p^2 + b p + c = grams``."""
    grams_all = force * 4.0 * 1000.0 / g
    discriminant = (b * b - 4.0 * a * (c - grams_all)).clamp(min=0.0)
    pwm = (-b + discriminant.sqrt()) / (2.0 * a)
    return pwm.clamp(0.0, pwm_max)


def motor_lag_step(thrust: torch.Tensor, thrust_cmd: torch.Tensor, tau_inc: torch.Tensor, tau_dec: torch.Tensor,
                   dt: float) -> torch.Tensor:
    """One step of the motor lag of Isaac Lab's ``Thruster`` (isaaclab_contrib) returns the new thrust (N, 4).

    The lag acts on the rotor speed, not on the thrust. Since T = k * eta^2, the speed is eta ~ sqrt(T).
    """
    speed = thrust.clamp(min=0.0).sqrt()               # eta       ~ sqrt(T)
    speed_cmd = thrust_cmd.clamp(min=0.0).sqrt()       # eta_cmd   ~ sqrt(T_cmd)

    # Time constant: tau_dec while the motor slows down, tau_inc while it speeds up.
    slowing = (thrust > 0.0) & (thrust_cmd < thrust)
    tau = torch.where(slowing, tau_dec, tau_inc)
    mixing = 1.0 / (dt + tau)                          # rate = error / (dt + tau)

    # RK4 on d(eta)/dt = mixing * e,  e = eta_cmd - eta, and e itself shrinks as eta grows during the step:
    #     k1 = r(e),  k2 = r(e - dt/2 k1),  k3 = r(e - dt/2 k2),  k4 = r(e - dt k3),   eta += dt/6 (k1 + 2 k2 + 2 k3 + k4)
    error = speed_cmd - speed
    k1 = mixing * error
    k2 = mixing * (error - 0.5 * dt * k1)
    k3 = mixing * (error - 0.5 * dt * k2)
    k4 = mixing * (error - dt * k3)
    speed = speed + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return speed * speed                               # back to thrust, T ~ eta^2


SPIN_VISUAL_SCALE = 0.06
"""Picture speed / rotor speed of the propellers: 86 rad/s on screen at hover instead of 1435 (see ``spin_propellers``)."""


def spin_propellers(robot, force: torch.Tensor, spin: torch.Tensor, scale: float) -> None:
    """Turn the propeller joints with the rotor speed, for the picture only (the wrench does the physics).

        eta_i    = sqrt(F_i / k_eta)                  rotor speed from F = k_eta eta^2 (about 1435 rad/s at hover)
        speed_i  = -s_i * scale * eta_i               s_i = spin sign of ``uav_cfg``; the joints turn opposite to it

    ``scale`` slows the picture down for two reasons: at 2 ms per step the true speed is 2.9 rad per step, close to half
    a turn, and the two blades would look still; and PhysX limits a joint to 100 rad/s, so a larger speed is cut anyway.
    Only the velocity is written: writing the joint angle as well froze the propellers after a few steps.
    """
    joint_ids = robot.find_joints(list(CF_MOTOR_JOINTS), preserve_order=True)[0]
    speed = -spin * scale * (force / CF_K_ETA).sqrt()
    robot.write_joint_velocity_to_sim_index(velocity=speed, joint_ids=joint_ids)


class Propulsion:
    """Motors of ``num_envs`` drones: holds their state and applies the wrench to the simulated body.

    ``cfg`` needs these fields (``MotorActionCfg`` has them): ``thrust_coef``, ``f_max``, ``gravity``, ``km``,
    ``motor_xy``, ``motor_spin``, ``use_motor_lag``, ``tau_inc_range``, ``tau_dec_range``, ``use_air_drag``,
    ``drag_coef``, ``drag_scale``, ``use_thrust_noise``, ``thrust_noise_std``, ``use_motor_asymmetry``,
    ``motor_strength_range``, ``spin_propellers``, ``spin_visual_scale``.
    """

    def __init__(self, cfg, num_envs: int, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Fixed model constants.
        self.a, self.b, self.c = cfg.thrust_coef
        xy = torch.tensor(cfg.motor_xy, dtype=torch.float32, device=device)     # (4, 2) motor positions (x, y)
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.spin = torch.tensor(cfg.motor_spin, dtype=torch.float32, device=device)   # (4,) +1 / -1

        # State, one row per environment.
        self.force = torch.zeros(num_envs, 4, device=device)      # F_i: thrust of each motor now [N]

        # Randomised every episode (see ``resample``).
        self.tau_inc = torch.zeros(num_envs, 4, device=device)    # lag while speeding up [s]
        self.tau_dec = torch.zeros(num_envs, 4, device=device)    # lag while slowing down [s]
        self.gain = torch.ones(num_envs, 4, device=device)        # g_i: strength of each motor
        self.drag = torch.zeros(num_envs, device=device)          # c_d: linear drag [N s/m]

        # Buffers written to the simulator (env, body, xyz).
        self._force_buf = torch.zeros(num_envs, 1, 3, device=device)
        self._torque_buf = torch.zeros(num_envs, 1, 3, device=device)


        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids) -> None:
        """Draw the motor lag, motor gain and drag of the environments starting a new episode."""
        cfg, n = self.cfg, len(env_ids)

        def uniform(lo_hi, *shape):
            lo, hi = lo_hi
            return lo + torch.rand(*shape, device=self.device) * (hi - lo)    # U(lo, hi)

        if cfg.use_motor_lag:
            self.tau_inc[env_ids] = uniform(cfg.tau_inc_range, n, 4)
            self.tau_dec[env_ids] = uniform(cfg.tau_dec_range, n, 4)
        else:
            self.tau_inc[env_ids] = 0.0
            self.tau_dec[env_ids] = 0.0
        self.gain[env_ids] = uniform(cfg.motor_strength_range, n, 4) if cfg.use_motor_asymmetry else 1.0
        self.drag[env_ids] = cfg.drag_coef * uniform(cfg.drag_scale, n) if cfg.use_air_drag else 0.0

    def reset(self, env_ids) -> None:
        """Motors at rest, new random parameters."""
        self.force[env_ids] = 0.0
        self.resample(env_ids)

    def step(self, pwm: torch.Tensor, robot, body_id, dt: float) -> None:
        """Advance the motors by ``dt`` [s] with the command ``pwm`` (N, 4) and apply the wrench to ``body_id``."""
        cfg = self.cfg

        # 1. Commanded thrust of each motor: curve at this PWM, scaled by the strength g_i of that motor.
        #        F_cmd_i = g_i * F(PWM_i)
        f_cmd = pwm_to_thrust(pwm, self.a, self.b, self.c, cfg.gravity, cfg.f_max) * self.gain

        # 2. Motor lag: the real thrust follows F_cmd with a delay.
        if cfg.use_motor_lag:
            self.force[:] = motor_lag_step(self.force, f_cmd, self.tau_inc, self.tau_dec, dt)
        else:
            self.force[:] = f_cmd
        f = self.force

        # 3. Thrust noise (optional):   F_i <- clip(F_i + N(0, (sigma * F_max)^2), 0, g_i * F_max)
        if cfg.use_thrust_noise and cfg.thrust_noise_std > 0.0:
            noisy = f + torch.randn_like(f) * (cfg.thrust_noise_std * cfg.f_max)
            f = torch.minimum(noisy.clamp(min=0.0), cfg.f_max * self.gain)

        # 4. Force on the body (body frame, z up).
        #        f = [0, 0, sum_i F_i] - c_d * v          v = velocity of the body in its own frame
        force = self._force_buf[:, 0]                                           # (N, 3) view of the buffer
        force[:, 0:2] = 0.0
        force[:, 2] = f.sum(dim=1)                                              # total thrust T = sum_i F_i
        if cfg.use_air_drag:
            force -= self.drag.unsqueeze(-1) * robot.data.root_lin_vel_b.torch  # linear drag  -c_d v

        # 5. Torque on the body.
        #        tau_x = sum_i F_i y_i          (a motor on the left, y > 0, pushes the left side up: roll +)
        #        tau_y = -sum_i F_i x_i         (a motor in front, x > 0, pushes the nose up: pitch -)
        #        tau_z = k_m sum_i s_i F_i      (reaction torque of the propeller drag, s_i = spin sign)
        torque = self._torque_buf[:, 0]                                         # (N, 3) view
        torque[:, 0] = (f * self.y).sum(dim=1)
        torque[:, 1] = -(f * self.x).sum(dim=1)
        torque[:, 2] = cfg.km * (f * self.spin).sum(dim=1)

        robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_id, forces=self._force_buf, torques=self._torque_buf)

        if cfg.spin_propellers:
            spin_propellers(robot, self.force, self.spin, self.cfg.spin_visual_scale)
