# Fichier de configuration modifie
# CAPITANO Giuliano

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLite3RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "deeprobotics_lite3_rough"
    empirical_normalization = False
    clip_actions = 100

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependant_std=False,
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        stochastic=False,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependant_std=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
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
class DeeproboticsLite3FlatPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10000
        self.experiment_name = "deeprobotics_lite3_flat"


@configclass
class CustomLite3LongJumpPPORunnerCft(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "custom_lite3_long_jump"
        self.algorithm.gamma = 0.96
        self.algorithm.entropy_coef = 0.03