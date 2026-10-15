"""Benchmark statistics from the per-step logs of one scenario set (pure numpy).

A log has one row per scenario ``s`` and one column per policy step ``k`` (the state after step ``k``, at time
``(k + 1) * dt``). ``valid[s, k]`` marks the steps that count: within the scenario's duration and up to (including) the
step where it failed, if it did. Distances in metres, angles in radians, margins as defined in ``LOG_FIELDS`` below.
"""

from __future__ import annotations

import numpy as np

# name: dimension of every logged per-step quantity (the runner stores each as (S, T) or (S, T, dim))
LOG_FIELDS = {
    "t": 1,  # time since the start of the scenario [s]
    "ref_pos": 3, "ref_quat": 4,  # reference pose, env-local, xyzw
    "ee_pos": 3, "ee_quat": 4,  # TCP pose, env-local, xyzw
    "pos_err_vec": 3,  # ee_pos - ref_pos [m]
    "pos_err": 1, "ori_err": 1,  # norm of the position error [m], angle between TCP and reference orientation [rad]
    "base_pos": 2, "yaw": 1, "roll": 1, "pitch": 1,  # base_link position (env-local), yaw/roll/pitch [rad]
    "base_vel": 2, "yaw_rate": 1,  # base velocity in the base frame [m/s], yaw rate [rad/s]
    "com_total": 3, "com_arm": 3,  # whole-robot and arm COM, env-local [m]
    "com_xy_platform": 2,  # whole-robot COM offset from the footprint centre in the platform frame (dx, dy) [m]
    "half_size": 2,  # half footprint length, width [m]
    "rho": 1,  # COM offset as a fraction of the half footprint: max(|dx| / half_length, |dy| / half_width); > 1 = outside
    "margin_m": 1,  # metres from the COM to the nearest footprint edge: min(half_length - |dx|, half_width - |dy|)
    "tipped": 1, "left_footprint": 1,  # latched failure flags (0/1)
    "arm_q": 7,  # arm joint positions [rad]
    "action": 3,  # policy action (vx, vy, wz) in [-1, 1]
}


def _stats(x: np.ndarray) -> dict:
    """Pooled statistics of the 1-D sample ``x`` (empty -> NaN)."""
    if not len(x):
        return {k: float("nan") for k in ("rms", "mean", "std", "p95", "max")}
    return {"rms": float(np.sqrt(np.mean(x**2))), "mean": float(x.mean()), "std": float(x.std()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _per_scenario(x: np.ndarray, valid: np.ndarray, reduce) -> np.ndarray:
    """``reduce(x[s, valid[s]])`` for every scenario with at least one valid step (NaN otherwise)."""
    return np.array([reduce(x[s][valid[s]]) if valid[s].any() else np.nan for s in range(len(x))])


def failure_steps(flags: np.ndarray, n_steps: np.ndarray) -> np.ndarray:
    """First step index with ``flags`` set within each scenario's own ``n_steps``, or -1 if it never fails.

    flags: (S, T) bool, tipped | left_footprint, latched by the env.
    """
    in_time = np.arange(flags.shape[1])[None, :] < n_steps[:, None]
    hit = flags & in_time
    return np.where(hit.any(1), hit.argmax(1), -1)


def valid_mask(n_steps: np.ndarray, fail_step: np.ndarray, n_columns: int) -> np.ndarray:
    k = np.arange(n_columns)[None, :]
    return (k < n_steps[:, None]) & ((fail_step[:, None] < 0) | (k <= fail_step[:, None]))


def pool_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate the scenarios of several logs (each a dict of the ``LOG_FIELDS`` arrays (S, T[, dim]) plus ``valid``,
    ``fail_step`` and ``n_steps``) into one, padding the shorter ones in time with invalid steps."""
    t_max = max(part["valid"].shape[1] for part in parts)
    out = {}
    for key in (*LOG_FIELDS, "valid"):
        padded = []
        for part in parts:
            x = part[key]
            pad = [(0, 0), (0, t_max - x.shape[1])] + [(0, 0)] * (x.ndim - 2)
            padded.append(np.pad(x, pad))  # zeros, and False for the valid mask
        out[key] = np.concatenate(padded)
    for key in ("fail_step", "n_steps"):
        out[key] = np.concatenate([part[key] for part in parts])
    return out


def summarize(log: dict[str, np.ndarray], valid: np.ndarray, fail_step: np.ndarray, dt: float,
              final_window_s: float = 0.5, n_steps: np.ndarray | None = None) -> dict:
    """Aggregate statistics of one set. ``log`` holds the ``LOG_FIELDS`` arrays (S, T[, dim]).

    With ``n_steps`` (steps per scenario) it also counts the scenarios that tipped / whose COM left the footprint at any
    time within their duration, including after the failure that ended the valid steps (the logs keep running)."""
    s_count = len(valid)
    failed = fail_step >= 0
    done = ~failed
    pos, ori = log["pos_err"][..., 0], log["ori_err"][..., 0]
    out = {
        "scenarios": int(s_count),
        "failed": int(failed.sum()),
        "failure_rate": float(failed.mean()),
        "failure_time_s": _stats((fail_step[failed] + 1) * dt)["mean"] if failed.any() else float("nan"),
    }
    if n_steps is not None:
        in_time = np.arange(valid.shape[1])[None, :] < n_steps[:, None]
        out["tipped"] = int(((log["tipped"][..., 0] > 0) & in_time).any(1).sum())
        out["left_footprint"] = int(((log["left_footprint"][..., 0] > 0) & in_time).any(1).sum())
    for name, err in (("pos_err_m", pos), ("ori_err_rad", ori)):
        rms = _per_scenario(err, valid, lambda v: float(np.sqrt(np.mean(v**2))))
        out[name] = {
            **_stats(err[valid]),  # all valid steps of all scenarios pooled
            "completed_rms": _stats(err[valid & done[:, None]])["rms"],  # pooled over scenarios that did not fail
            "per_scenario_rms": {"mean": float(np.nanmean(rms)), "std": float(np.nanstd(rms)),
                                 "median": float(np.nanmedian(rms)), "max": float(np.nanmax(rms))},
        }
    # tracking error at the end of the scenario (last ``final_window_s``), for scenarios that completed
    last = valid.sum(1)
    window = int(round(final_window_s / dt))
    final = [pos[s, max(last[s] - window, 0): last[s]].mean() for s in range(s_count) if done[s] and last[s] > 0]
    out["final_pos_err_m"] = {"mean": float(np.mean(final)) if final else float("nan"),
                              "max": float(np.max(final)) if final else float("nan")}
    margin = log["margin_m"][..., 0]
    rho = log["rho"][..., 0]
    min_margin = _per_scenario(margin, valid, np.min)
    out["com_margin_m"] = {"mean": float(margin[valid].mean()), "std": float(margin[valid].std()),
                           "min": float(np.nanmin(min_margin)), "per_scenario_min_mean": float(np.nanmean(min_margin)),
                           "per_scenario_min_p5": float(np.nanpercentile(min_margin, 5))}
    out["com_rho"] = {"mean": float(rho[valid].mean()), "std": float(rho[valid].std()), "max": float(rho[valid].max())}
    speed = np.linalg.norm(log["base_vel"], axis=-1)
    out["base"] = {"speed_mean_mps": float(speed[valid].mean()), "speed_max_mps": float(speed[valid].max()),
                   "yaw_rate_abs_mean": float(np.abs(log["yaw_rate"][..., 0])[valid].mean()),
                   "max_tilt_rad": float(np.maximum(np.abs(log["roll"][..., 0]), np.abs(log["pitch"][..., 0]))[valid].max()),
                   "max_distance_from_spawn_m": float(np.linalg.norm(log["base_pos"], axis=-1)[valid].max())}
    return out


def format_table(summaries: dict[str, dict]) -> str:
    """One line per set: failures, RMS / std of the position error, orientation RMS, COM margin."""
    head = (f"{'set':10s} {'n':>4s} {'fail%':>6s} {'pos rms':>8s} {'pos std':>8s} {'pos p95':>8s} {'ori rms':>8s} "
            f"{'final':>7s} {'rho mean':>8s} {'rho max':>8s} {'min margin':>10s}")
    units = (f"{'':10s} {'':>4s} {'':>6s} {'[cm]':>8s} {'[cm]':>8s} {'[cm]':>8s} {'[deg]':>8s} "
             f"{'[cm]':>7s} {'':>8s} {'':>8s} {'[cm]':>10s}")
    lines = [head, units]
    for name, s in summaries.items():
        p, o = s["pos_err_m"], s["ori_err_rad"]
        lines.append(
            f"{name:10s} {s['scenarios']:4d} {100 * s['failure_rate']:6.1f} {100 * p['rms']:8.2f} {100 * p['std']:8.2f} "
            f"{100 * p['p95']:8.2f} {np.degrees(o['rms']):8.2f} {100 * s['final_pos_err_m']['mean']:7.2f} "
            f"{s['com_rho']['mean']:8.3f} {s['com_rho']['max']:8.3f} {100 * s['com_margin_m']['min']:10.2f}")
    return "\n".join(lines)
