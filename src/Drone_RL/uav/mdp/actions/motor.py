"""Plot the thrust of one Crazyflie 2.1 Brushless motor, following Folk (arXiv:2604.00343), section 6.3.1, fig. 6.5.

    python src/Drone_RL/uav/mdp/actions/motor.py                       # writes guide/media/motor_thrust.png
    python src/Drone_RL/uav/mdp/actions/motor.py --output thrust.png
    python src/Drone_RL/uav/mdp/actions/motor.py --show                # open a window instead

The paper identified the motor on a thrust stand:
    thrust     T = K_ETA * eta^2                                   (eta = rotor speed [rad/s])
    rotor speed eta = KV * (V + V0) * (PWM - DZ)^(2/3)              (V = battery voltage [V], PWM 16-bit)
Panels 1 and 2 redraw the two fits of fig. 6.5, panel 3 combines them (thrust vs PWM per battery voltage) and adds the
curve the simulation uses: a quadratic fit of the model at the nominal voltage (``uav_cfg.CF_THRUST_COEF_G``). The
constants live in ``uav_cfg.py``; this file only plots.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "cm"

from Drone_RL.uav import uav_cfg as U

G = 9.80665

K_ETA, KV, V0, DZ = U.CF_K_ETA, U.CF_KV, U.CF_V0, U.CF_DZ

"""PWM range covered by the experiment. Outside it the curves are extrapolated."""
PWM_MEASURED = (10_000.0, 45_000.0)

"""Battery voltages drawn [V]; the experiment covered 2.7 to 4.1 V. The simulation uses the nominal one."""
VOLTAGES = (2.7, 3.3, U.CF_BATTERY_V_NOM, 4.1)


def rotor_speed(pwm: np.ndarray, voltage: float) -> np.ndarray:
    """Rotor speed [rad/s] from the PWM command and the battery voltage."""
    return KV * (voltage + V0) * np.clip(pwm - DZ, 0.0, None) ** (2.0 / 3.0)


def thrust_from_speed(speed: np.ndarray) -> np.ndarray:
    """Thrust of one rotor [N] from the rotor speed [rad/s]."""
    return K_ETA * speed**2


def simulated_thrust_n(pwm: np.ndarray) -> np.ndarray:
    """Thrust of one motor [N] from the quadratic fit the simulation uses (nominal battery voltage)."""
    a, b, c = U.CF_THRUST_COEF_G
    return np.clip((a * pwm**2 + b * pwm + c) / 4.0 * G / 1000.0, 0.0, None)


def plot_motor_thrust(axes) -> None:
    """Draw the three panels on ``axes`` (a sequence of three matplotlib axes)."""
    ax_speed, ax_pwm, ax_thrust = axes
    colors = dict(zip(VOLTAGES, ("#1565c0", "#00838f", "#ef6c00", "#c62828")))
    pwm = np.linspace(DZ, U.CF_PWM_MAX, 400)
    inside = (pwm >= PWM_MEASURED[0]) & (pwm <= PWM_MEASURED[1])

    eta = np.linspace(0.0, 2100.0, 200)
    ax_speed.plot(eta, thrust_from_speed(eta), color="k", linestyle="--")
    ax_speed.set_xlabel(r"Motor speed $\eta$ (rad/s)")
    ax_speed.set_ylabel(r"Thrust $T$ (N)")
    ax_speed.set_title(rf"$T = {K_ETA * 1e8:.3f}\times10^{{-8}}\,\eta^{{2}}$", fontsize=11)

    for volt in VOLTAGES:
        speed = rotor_speed(pwm, volt)
        ax_pwm.plot(pwm[inside], speed[inside], color=colors[volt], label=rf"$V_s$ = {volt} V")
        ax_pwm.plot(pwm[~inside], speed[~inside], color=colors[volt], linestyle=":")
        thrust = thrust_from_speed(speed)
        ax_thrust.plot(pwm[inside], thrust[inside], color=colors[volt], label=rf"$V_s$ = {volt} V")
        ax_thrust.plot(pwm[~inside], thrust[~inside], color=colors[volt], linestyle=":")
    ax_pwm.set_xlabel("Motor PWM (16-bit int)")
    ax_pwm.set_ylabel(r"Motor speed $\eta$ (rad/s)")
    ax_pwm.set_title(rf"$\eta = {KV}\,(V_s + {V0})\,(\mathrm{{PWM}} - {DZ:.0f})^{{2/3}}$", fontsize=11)
    ax_pwm.legend(frameon=False, fontsize=8)

    ax_thrust.plot(pwm, simulated_thrust_n(pwm), color="k", linestyle="--", linewidth=2.0, label="simulation")
    hover_n = U.DRONE_HOVER_THRUST_N / 4.0
    ax_thrust.plot([U.DRONE_HOVER_THROTTLE * U.CF_PWM_MAX], [hover_n], "o", color="#2e7d32")
    ax_thrust.annotate(
        rf"hover: $T$ = {hover_n:.3f} N ({hover_n / G * 1000:.1f} g)",
        (U.DRONE_HOVER_THROTTLE * U.CF_PWM_MAX, hover_n),
        textcoords="offset points",
        xytext=(10, -22),
        fontsize=8,
        color="#2e7d32",
    )
    ax_thrust.axhline(U.CF_F_MAX_N, color="#616161", linewidth=0.8, linestyle=":")
    ax_thrust.set_xlabel("Motor PWM (16-bit int)")
    ax_thrust.set_ylabel(r"Thrust $T$ (N)")
    ax_thrust.set_title(r"$T(\mathrm{PWM}, V_s)$, dotted: outside 10k to 45k", fontsize=11)
    ax_thrust.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)
        ax.set_xlim(left=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("guide/media/motor_thrust.png"))
    parser.add_argument("--show", action="store_true", help="Open a window instead of writing a file.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    plot_motor_thrust(axes)
    fig.suptitle("Crazyflie 2.1 Brushless motor (Folk, arXiv:2604.00343, fig. 6.5)", fontweight="bold")
    if args.show:
        plt.show()
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    print(f"Image: {args.output}")


# if __name__ == "__main__":
#     main()
