"""Propulsion of the Crazyflie: PWM of the four motors -> force and torque on the body.

This file is the simulated stand-in for the hardware below the controller: on the real drone the ESCs, motors,
propellers and the air turn the PWM into forces and nobody computes them; here they have to be computed. 

    PWM  --pwm_to_thrust-->  F  --(noise)-->  F  --wrench-->  force, torque on the body
    (controller)             thrust of         the world        what PhysX integrates
                             each motor        adds             (``sim.step``, 2 ms)
                                               (turbulence)

The controller (or the policy) only writes a new PWM at its own rate and the value is held in between.

The opposite direction (wanted thrust and torques -> force of each motor -> PWM) belongs to the controller:
``mixer.force_allocation_inverse`` and ``thrust_to_pwm`` below. Only the PID rate layer uses it.
"""
import torch

from Drone_RL.uav.uav_cfg import CF_K_ETA, CF_MOTOR_JOINTS


def pwm_to_thrust(pwm: torch.Tensor, a: float, b: float, c: float, g: float, f_max: float) -> torch.Tensor:
    """PWM (N, 4) -> thrust of each motor (N, 4) [N]. The static thrust curve.

    Real counterpart: the thrust stand measurement of the motor + propeller (Folk et al., guide 1.2).

    ``a*pwm^2 + b*pwm + c`` (``uav_cfg.CF_THRUST_COEF_G``) is the thrust of the four motors together, in grams, so the
    per-motor value is a quarter of it (a per-motor fit made the hover throttle saturate at 1.0).

        pwm      0 .. 65535 (``CF_PWM_MAX``), the 16-bit motor command of the firmware
        g        gravity [m/s^2], only to turn grams into newtons
        f_max    thrust of one motor at full PWM [N] (``CF_F_MAX_N``)
    """
    grams_all = a * pwm * pwm + b * pwm + c        # total thrust [g], all four motors at this PWM
    newton_one = grams_all / 4.0 * g / 1000.0      # one motor: / 4, then grams -> kg (/ 1000), then kg -> N (* g)
    return newton_one.clamp(0.0, f_max)            # no negative thrust below the dead zone, cap at full command


def thrust_to_pwm(force: torch.Tensor, a: float, b: float, c: float, g: float, pwm_max: float) -> torch.Tensor:
    """Inverse of ``pwm_to_thrust``: thrust of each motor (N, 4) [N] -> PWM (N, 4). Solves ``a p^2 + b p + c = grams``.

    Controller side, not physics: the PID rate layer and the feedforward of the RL rate layer use it to ask a motor for
    a thrust. Kept here so the curve and its inverse live next to each other and cannot drift apart.
    """
    grams_all = force * 4.0 * 1000.0 / g                                # one motor [N] -> all four [g]
    discriminant = (b * b - 4.0 * a * (c - grams_all)).clamp(min=0.0)   # < 0 only below the dead zone: give PWM 0 there
    pwm = (-b + discriminant.sqrt()) / (2.0 * a)                        # the positive root of the quadratic
    return pwm.clamp(0.0, pwm_max)


SPIN_VISUAL_SCALE = 0.06
"""Picture speed / rotor speed of the propellers: 99 rad/s on screen at hover instead of 1650 (see ``spin_propellers``)."""


def spin_propellers(robot, force: torch.Tensor, spin: torch.Tensor, scale: float) -> None:
    """Turn the propeller joints with the rotor speed, for the picture only (the wrench does the physics).

        eta_i    = sqrt(F_i / k_eta)                  rotor speed from F = k_eta eta^2 (about 1650 rad/s at hover)
        speed_i  = -s_i * scale * eta_i               s_i = spin sign of ``uav_cfg``; the joints turn opposite to it

    ``scale`` slows the picture down for two reasons: at 2 ms per step the true speed is 3.3 rad per step, close to half
    a turn, and the two blades would look still; and PhysX limits a joint to 100 rad/s, so a larger speed is cut anyway.
    Only the velocity is written: writing the joint angle as well froze the propellers after a few steps.
    """
    joint_ids = robot.find_joints(list(CF_MOTOR_JOINTS), preserve_order=True)[0]
    speed = -spin * scale * (force / CF_K_ETA).sqrt()
    robot.write_joint_velocity_to_sim_index(velocity=speed, joint_ids=joint_ids)


class Propulsion:
    """Motors of ``num_envs`` drones: holds their state and applies the wrench to the simulated body.

    ``cfg`` needs these fields (``MotorActionCfg`` has them): ``thrust_coef``, ``f_max``, ``gravity``, ``km``,
    ``motor_xy``, ``motor_spin``, ``use_air_drag``,
    ``drag_coef``, ``drag_scale``, ``use_thrust_noise``, ``thrust_noise_std``, ``use_motor_asymmetry``,
    ``motor_strength_range``, ``spin_propellers``, ``spin_visual_scale``.
    """

    def __init__(self, cfg, num_envs: int, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Fixed model constants.
        self.a, self.b, self.c = cfg.thrust_coef                                # thrust curve of pwm_to_thrust
        xy = torch.tensor(cfg.motor_xy, dtype=torch.float32, device=device)     # (4, 2) motor positions (x, y)
        self.x, self.y = xy[:, 0], xy[:, 1]
        self.spin = torch.tensor(cfg.motor_spin, dtype=torch.float32, device=device)   # (4,) +1 / -1

        # State, one row per environment.
        self.force = torch.zeros(num_envs, 4, device=device)      # F_i: thrust of each motor now [N]

        # Randomised every episode (see ``resample``): the real drone differs from the model, so the controller is
        # trained or tested on a spread of drones rather than on one exact copy (domain randomisation).
        self.gain = torch.ones(num_envs, 4, device=device)        # g_i: strength of each motor
        self.drag = torch.zeros(num_envs, device=device)          # c_d: linear drag [N s/m]

        # Buffers written to the simulator (env, body, xyz): one body (``body``), force and torque in the body frame.
        self._force_buf = torch.zeros(num_envs, 1, 3, device=device)
        self._torque_buf = torch.zeros(num_envs, 1, 3, device=device)


        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids) -> None:
        """Draw the motor gain and drag of the environments starting a new episode."""
        cfg, n = self.cfg, len(env_ids)

        def uniform(lo_hi, *shape):
            lo, hi = lo_hi
            return lo + torch.rand(*shape, device=self.device) * (hi - lo)    # U(lo, hi)

        self.gain[env_ids] = uniform(cfg.motor_strength_range, n, 4) if cfg.use_motor_asymmetry else 1.0
        self.drag[env_ids] = cfg.drag_coef * uniform(cfg.drag_scale, n) if cfg.use_air_drag else 0.0

    def reset(self, env_ids) -> None:
        """Motors at rest, new random parameters."""
        self.force[env_ids] = 0.0
        self.resample(env_ids)

    def step(self, pwm: torch.Tensor, robot, body_id) -> None:
        """Turn the command ``pwm`` (N, 4) into the wrench on ``body_id``.

        Called once per physics step (1 kHz), before ``sim.step``, by ``MotorAction`` / ``CascadeAction.apply_actions``
        in RL and by hand in the PID scripts. It does not move the drone itself: it only sets the force and torque, and
        PhysX integrates them in the following ``sim.step``.
        """
        cfg = self.cfg

        # 1. Thrust of each motor: curve at this PWM, scaled by the strength g_i of that motor.
        #        F_i = g_i * F(PWM_i)
        #    g_i (0.9 .. 1.1) stands for the spread between real motors and propellers: the same PWM never gives exactly
        #    the same thrust on all four, and the controller has to correct it from the gyro. Kept in ``self.force``
        #    for ``CascadeAction.thrust`` and the propeller picture.
        self.force[:] = pwm_to_thrust(pwm, self.a, self.b, self.c, cfg.gravity, cfg.f_max) * self.gain
        f = self.force

        # 2. Thrust noise (optional):   F_i <- clip(F_i + N(0, (sigma * F_max)^2), 0, g_i * F_max)
        #    Turbulence and vibration that no model predicts, sigma = 1 % of the full thrust. It is applied to the copy
        #    ``f`` only; ``self.force`` keeps the thrust without noise.
        if cfg.use_thrust_noise and cfg.thrust_noise_std > 0.0:
            noisy = f + torch.randn_like(f) * (cfg.thrust_noise_std * cfg.f_max)
            f = torch.minimum(noisy.clamp(min=0.0), cfg.f_max * self.gain)

        # 3. Force on the body (body frame, z up). This and step 4 are the FORWARD direction (four forces -> one
        #    wrench); on the real drone the frame does this by itself, here it has to be written out. Matrix form:
        #    ``mixer.wrench_matrix``; its inverse is what the PID rate controller uses.
        #        f = [0, 0, sum_i F_i] - c_d * v          v = velocity of the body in its own frame
        #    All propellers push along the body z axis, so x and y of the thrust are 0. The drag term -c_d v is the
        #    induced drag of the rotors (Folk et al., guide 2.3.3); PhysX itself computes no air at all.
        force = self._force_buf[:, 0]                                           # (N, 3) view of the buffer
        force[:, 0:2] = 0.0
        force[:, 2] = f.sum(dim=1)                                              # total thrust T = sum_i F_i
        if cfg.use_air_drag:
            force -= self.drag.unsqueeze(-1) * robot.data.root_lin_vel_b.torch  # linear drag  -c_d v

        # 4. Torque on the body: torque = r x F of each motor about the centre of mass, plus the propeller drag.
        #        tau_x = sum_i F_i y_i          (a motor on the left, y > 0, pushes the left side up: roll +)
        #        tau_y = -sum_i F_i x_i         (a motor in front, x > 0, pushes the nose up: pitch -)
        #        tau_z = k_m sum_i s_i F_i      (reaction torque of the propeller drag, s_i = spin sign)
        #    s_i = +1 for a propeller turning clockwise seen from above: the air drags it back, so the frame turns the
        #    other way (yaw +). At hover the four F_i are equal and the +1 and -1 pairs cancel. k_m = drag torque /
        #    thrust of one propeller (about 2 mm), which is why yaw authority is much weaker than roll and pitch.
        torque = self._torque_buf[:, 0]                                         # (N, 3) view
        torque[:, 0] = (f * self.y).sum(dim=1)
        torque[:, 1] = -(f * self.x).sum(dim=1)
        torque[:, 2] = cfg.km * (f * self.spin).sum(dim=1)

        # Hand the wrench to Isaac Lab. "Permanent": it stays applied at every PhysX step until it is overwritten (here,
        # at the next call). It is only stored now; ``Articulation.write_data_to_sim`` passes it to PhysX
        # (``apply_forces_and_torques_at_position``, body frame, at the centre of mass) just before ``sim.step``.
        # Gravity, the gyroscopic term and the ground contact are added by PhysX itself (guide 7.6).
        robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=body_id, forces=self._force_buf, torques=self._torque_buf)

        # Picture only: PhysX does not compute any lift from the spinning propellers, the wrench above does the physics.
        if cfg.spin_propellers:
            spin_propellers(robot, self.force, self.spin, self.cfg.spin_visual_scale)
