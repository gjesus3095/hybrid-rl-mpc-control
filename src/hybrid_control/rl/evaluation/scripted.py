"""Table-driven reference trajectories: a drop-in for ``ReferenceTrajectories`` that plays pre-generated scenarios.

Env slot ``j`` plays scenario ``j`` of a ``ScenarioSet`` (see ``scenarios.py``). Nothing is sampled at a reset: the env's
reset only checks that the simulated spawn pose is the one the scenarios were generated for. Pure torch, no Isaac imports.
"""

from __future__ import annotations

import numpy as np
import torch

from hybrid_control.rl.policy_with_controller.trajectories import ReferenceTrajectories, TrajectoryCfg
from hybrid_control.utils.torch_rotations import quat_angle

from .scenarios import REF_DT, ScenarioSet

SPAWN_TOLERANCE = (2e-3, 0.02)  # position [m] and orientation [rad] by which the simulated spawn may differ


def combine_scenarios(entries: list[tuple[ScenarioSet, int]], name: str = "mixed") -> ScenarioSet:
    """One ``ScenarioSet`` from scenario ``index`` of several sets (``entries`` = [(set, index)]), e.g. to replay the
    failures of different sets together. Shorter tables hold their last pose up to the longest one."""
    t_len = max(s.poses.shape[1] for s, _ in entries)
    poses = np.stack([np.concatenate([s.poses[i], np.repeat(s.poses[i, -1:], t_len - s.poses.shape[1], axis=0)])
                      for s, i in entries])
    pick = lambda key: np.stack([getattr(s, key)[i] for s, i in entries])  # noqa: E731
    return ScenarioSet(name, poses, pick("duration"), pick("centre"), pick("dynamics"))


class ScriptedTrajectories(ReferenceTrajectories):
    def __init__(self, num_envs: int, device: str | torch.device, cfg: TrajectoryCfg | None = None):
        super().__init__(num_envs, device, cfg)
        self.name = "scripted"
        self.table = torch.zeros(num_envs, 2, 7, device=device)
        self.table[..., 6] = 1.0
        self.scenario_duration = torch.zeros(num_envs, device=device)

    def load(self, scenario_set: ScenarioSet, scenario_ids=None):
        """Play scenario ``scenario_ids[j]`` of ``scenario_set`` in env slot ``j`` (default: scenario ``j``; the set
        must provide ``num_envs`` scenarios)."""
        ids = np.arange(self.num_envs) if scenario_ids is None else np.asarray(scenario_ids)
        if len(ids) != self.num_envs or ids.max() >= len(scenario_set):
            raise ValueError(f"set {scenario_set.name!r} has {len(scenario_set)} scenarios for {self.num_envs} env slots")
        self.name = scenario_set.name
        self.table = torch.as_tensor(scenario_set.poses[ids], dtype=torch.float32, device=self.device)
        self.scenario_duration = torch.as_tensor(scenario_set.duration[ids], dtype=torch.float32, device=self.device)
        self.centre = torch.as_tensor(scenario_set.centre[ids], dtype=torch.float32, device=self.device)

    def reset(self, env_ids, start, start_quat=None, kind=None):
        """Keeps the loaded scenarios; raises if the simulated spawn pose is not the one the scenarios start from."""
        env_ids = env_ids.to(self.device)
        start = start.to(self.device, torch.float32)
        pos_err = (start - self.table[env_ids, 0, :3]).norm(dim=-1).max()
        ori_err = 0.0
        if start_quat is not None:
            ori_err = quat_angle(start_quat.to(self.device, torch.float32), self.table[env_ids, 0, 3:]).max()
        if pos_err > SPAWN_TOLERANCE[0] or ori_err > SPAWN_TOLERANCE[1]:
            raise ValueError(f"simulated spawn pose differs from the benchmark's by {float(pos_err):.4f} m / "
                             f"{float(ori_err):.4f} rad: update SPAWN_POS / SPAWN_QUAT in scenarios.py and regenerate")

    def kind_name(self, i: int) -> str:
        return self.name

    def phase_name(self, i: int, t: float) -> str:
        return "following" if t < float(self.scenario_duration[i]) else "holding end"

    def lookahead_pose(self, t: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference positions (N, K, 3) and orientations (N, K, 4, xyzw) at times ``t + offsets`` (K,): the table is
        interpolated linearly (normalized lerp for the quaternions) and the last pose is held past its end."""
        last = self.table.shape[1] - 1
        x = ((t[:, None] + offsets[None, :]) / REF_DT).clamp(0.0, float(last))
        i0 = x.floor().long().clamp(max=last - 1)
        frac = (x - i0)[..., None]
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        p0, p1 = self.table[rows, i0], self.table[rows, i0 + 1]
        q0, q1 = p0[..., 3:], p1[..., 3:]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        quat = q0 + frac * (q1 - q0)
        return p0[..., :3] + frac * (p1[..., :3] - p0[..., :3]), quat / quat.norm(dim=-1, keepdim=True)
