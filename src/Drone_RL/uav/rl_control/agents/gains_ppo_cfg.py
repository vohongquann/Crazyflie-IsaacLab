"""PPO (rsl_rl) settings of the gain tasks (``Isaac-UAV-<Layer>-Gains-v0``).

Actor 64 x 64 (small enough for the STM32 of the Crazyflie), observation normalisation inside the actor so the frozen
network carries it along. The gains act through a PID, so the noise of a gain (the std is in the exponent of 3^a) stays
small: 0.3 at most. Each task has its own log folder ``uav_<layer>_gains``.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def experiment_name(layer: str) -> str:
    return f"uav_{layer}_gains"


@configclass
class BoundedGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian action noise whose std is clamped (rsl_rl ``GaussianDistribution.std_range``). Without a bound the std
    can grow until the actions sit at the clip limits."""

    std_range: tuple[float, float] = (0.02, 0.3)


def gains_actor() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianCfg(init_std=0.2),
    )


@configclass
class GainsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 50
    clip_actions = 1.0
    experiment_name = experiment_name("velocity")
    actor = gains_actor()
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,          # the action noise only shrinks as the policy learns
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RateGainsPPORunnerCfg(GainsPPORunnerCfg):
    num_steps_per_env = 64
    max_iterations = 1500
    experiment_name = experiment_name("rate")


@configclass
class AttitudeGainsPPORunnerCfg(GainsPPORunnerCfg):
    num_steps_per_env = 48
    max_iterations = 1000
    experiment_name = experiment_name("attitude")


@configclass
class VelocityGainsPPORunnerCfg(GainsPPORunnerCfg):
    experiment_name = experiment_name("velocity")


@configclass
class PositionGainsPPORunnerCfg(GainsPPORunnerCfg):
    max_iterations = 1000
    experiment_name = experiment_name("position")
