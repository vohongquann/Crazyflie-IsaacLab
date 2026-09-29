"""PPO (rsl_rl) settings shared by the four RL cascade tasks; each layer file only changes what differs.

Actor 64 x 64 (small enough for the STM32 of the Crazyflie), observation normalisation inside the actor so the frozen
network carries it along. Iterations are for 1024 environments (the number used with ``--video``). The experiment name ``uav_<layer>`` is where ``freeze.py`` looks for the last checkpoint.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from Drone_RL.uav.rl_control.freeze import experiment_name


@configclass
class BoundedGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Gaussian action noise whose std is clamped (rsl_rl ``GaussianDistribution.std_range``). Without a bound the std
    of the first rate run grew from 0.2 to 8.5: the actions sat at the clip limits and the motors only switched."""

    std_range: tuple[float, float] = (0.02, 0.5)


def make_actor(init_std: float, std_range: tuple[float, float] = (0.02, 0.5)) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BoundedGaussianCfg(init_std=init_std, std_range=std_range),
    )


@configclass
class LayerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 50
    clip_actions = 1.0
    experiment_name = experiment_name("position")
    actor = make_actor(init_std=0.5)
    critic = RslRlMLPModelCfg(hidden_dims=[128, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,          # the action noise only shrinks as the policy learns (0.002 kept it at its cap)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
