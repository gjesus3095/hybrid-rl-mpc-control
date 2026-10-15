"""Strict configuration loading for moving-mass checkpoints."""
from dataclasses import fields
from pathlib import Path
import yaml

from hybrid_control.rl.config import make_runner_cfg
from hybrid_control.sim_interfaces.isaac.moving_mass.params import MechanismCfg, MovingMassRandomization, PlatformRandomization
from .env import StabilizationEnvCfg
from .observations import ObservationCfg
from .rewards import RewardCfg
from .trajectories import TrajectoryCfg


def _construct(cls, values):
    values = dict(values)
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'unknown {cls.__name__} fields: {sorted(unknown)}')
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def load_env_cfg(values):
    values = dict(values)
    if values.get('environment_kind') != 'moving_mass_v3':
        raise ValueError('checkpoint/config is not a moving_mass_v3 environment; older planar/radial moving-mass and Panda checkpoints are incompatible')
    for key, cls in [('mechanism', MechanismCfg), ('trajectory', TrajectoryCfg),
                     ('observation', ObservationCfg), ('reward', RewardCfg)]:
        if key in values:
            values[key] = _construct(cls, values[key])
    if 'domain_randomization' in values:
        dr = dict(values['domain_randomization'])
        if 'platform' in dr:
            dr['platform'] = _construct(PlatformRandomization, dr['platform'])
        values['domain_randomization'] = _construct(MovingMassRandomization, dr)
    return _construct(StabilizationEnvCfg, values)


def load_configuration(config_path=None, checkpoint=None):
    """Resume/play always restores the checkpoint's entire saved configuration."""
    if checkpoint and config_path:
        raise ValueError('do not override configuration when loading a checkpoint')
    if checkpoint:
        directory = Path(checkpoint).resolve().parent / 'params'
        env_path, agent_path = directory/'env.yaml', directory/'agent.yaml'
        if not env_path.is_file() or not agent_path.is_file():
            raise ValueError('moving-mass checkpoints require params/env.yaml and params/agent.yaml')
        return load_env_cfg(yaml.safe_load(env_path.read_text())), yaml.safe_load(agent_path.read_text())
    cfg = load_env_cfg(yaml.safe_load(Path(config_path).read_text())) if config_path else StabilizationEnvCfg()
    return cfg, make_runner_cfg()


def create_environment(sim, cfg, num_envs, spacing, seed):
    # Import the simulator backend only after entering launch_simulation.
    from hybrid_control.sim_interfaces.isaac.moving_mass.env import MovingMassEnv
    from .env import BaseStabilizationEnv
    base = MovingMassEnv(cfg.domain_randomization.sample_many(num_envs, seed), sim, spacing=spacing,
                         friction=cfg.friction, tau=cfg.base_time_constant, max_force=cfg.max_base_force,
                         mechanism=cfg.mechanism)
    base.start()
    return BaseStabilizationEnv(base, cfg, seed)
