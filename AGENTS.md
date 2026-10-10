# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains documentation and an empty directory scaffold. The README lists planned files; source code, package metadata, and configuration files are not implemented yet.

- `src/hybrid_control/controllers/`: separate `ik_qp/` manipulation and `rl/` base controllers.
- `src/hybrid_control/robots/`: shared robot models and state representations.
- `src/hybrid_control/sim_interfaces`: MuJoCo and Isaac adapters, with `mujoco/models/` and `isaac/assets/` for simulator resources.
- `src/hybrid_control/environments/isaac_lab/`: planned training environment configuration, observations, rewards, actions, and terminations.
- `src/hybrid_control/integration/` and `utils/`: controller coordination and reusable mathematical helpers.
- `configs/`, `scripts/`, `tests/`, and `docs/`: configuration, simulator-specific entry points, automated tests, and design notes.

Empty directories are not tracked by Git. Add implementation files only when required by the task.

## Build, Test, and Development Commands

No build, installation, simulation, or test commands are configured yet. Do not assume the planned `pyproject.toml` or scripts exist.

- `git status --short`: review changed and untracked files.
- `git diff --check`: check tracked changes for whitespace errors.
- `git diff -- README.md`: review project documentation edits.

Document reproducible setup and execution commands when adding the first implementation.

## Coding Style & Naming Conventions

For planned Python modules, use four-space indentation, `snake_case` for modules and functions, and `PascalCase` for classes. Keep simulator-specific code in its backend or environment package. Document coordinate frames, physical units, and array shapes at controller interfaces. No formatter or linter is configured yet.

## Testing Guidelines

No testing framework or coverage threshold currently exists. Place future automated tests in `tests/` using `test_*.py` names. Cover kinematics, IK/QP constraints, and hybrid controller interactions; document simulator dependencies and random seeds for reproducibility.

## Commit & Pull Request Guidelines

Existing history uses short, descriptive subjects such as `Revise README for Hybrid RL-IK Control project`; no mandatory prefix convention is established. Keep commits focused. Pull requests should explain the change, link relevant issues, and state validation performed or unavailable. Include tracking plots or simulation evidence when changing controller behavior.
