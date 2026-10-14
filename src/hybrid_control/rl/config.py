"""RSL-RL (5.x) PPO runner configuration as a plain dict, so no Isaac Lab config classes are needed."""

from __future__ import annotations


def make_runner_cfg(
    num_steps_per_env: int = 24,
    save_interval: int = 50,
    hidden_dims: tuple[int, ...] = (256, 256, 128),
    init_std: float = 0.5,
    learning_rate: float = 1.0e-3,
    logger: str = "tensorboard",
) -> dict:
    """PPO with an MLP actor/critic; the observation groups ``actor`` and ``critic`` come from the env."""
    return {
        "num_steps_per_env": num_steps_per_env,
        "save_interval": save_interval,
        "check_for_nan": True,
        "logger": logger,
        "multi_gpu": None,
        "torch_compile_mode": None,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": init_std, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(hidden_dims),
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": learning_rate,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.005,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "use_mixed_precision": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }
