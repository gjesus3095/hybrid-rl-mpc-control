"""One readable Markdown report of an ``evaluate_policy.py --payloads ...`` run (pure numpy, no Isaac imports).

Every table has one row per metric and one column per trajectory type, plus an ``ALL`` column that pools every
scenario. Pooled statistics (RMS, std, percentiles, ...) are recomputed from the per-step logs, not averaged from
the per-set summaries. Sections: everything pooled over all payloads (and a comparison by payload), then one table per
payload, in ascending order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import LOG_FIELDS, pool_parts, summarize
from .scenarios import SET_NAMES

DEG = float(np.degrees(1.0))

# (section, label, getter(summary) -> value, format). ``None`` rows are section headers.
ROWS = [
    ("Outcome", None, None, None),
    ("", "scenarios", lambda s: s["scenarios"], "{:d}"),
    ("", "failed: COM left footprint or tipped (n)", lambda s: s["failed"], "{:d}"),
    ("", "failure rate [%]", lambda s: 100 * s["failure_rate"], "{:.1f}"),
    ("", "mean time of failure [s]", lambda s: s["failure_time_s"], "{:.2f}"),
    ("", "tipped at any time (n)", lambda s: s["tipped"], "{:d}"),
    ("", "COM left footprint at any time (n)", lambda s: s["left_footprint"], "{:d}"),
    ("Position error (TCP vs reference)", None, None, None),
    ("", "RMS [cm]", lambda s: 100 * s["pos_err_m"]["rms"], "{:.2f}"),
    ("", "mean [cm]", lambda s: 100 * s["pos_err_m"]["mean"], "{:.2f}"),
    ("", "std [cm]", lambda s: 100 * s["pos_err_m"]["std"], "{:.2f}"),
    ("", "95th percentile [cm]", lambda s: 100 * s["pos_err_m"]["p95"], "{:.2f}"),
    ("", "max [cm]", lambda s: 100 * s["pos_err_m"]["max"], "{:.2f}"),
    ("", "RMS, completed scenarios only [cm]", lambda s: 100 * s["pos_err_m"]["completed_rms"], "{:.2f}"),
    ("", "per-scenario RMS: mean [cm]", lambda s: 100 * s["pos_err_m"]["per_scenario_rms"]["mean"], "{:.2f}"),
    ("", "per-scenario RMS: std [cm]", lambda s: 100 * s["pos_err_m"]["per_scenario_rms"]["std"], "{:.2f}"),
    ("", "per-scenario RMS: median [cm]", lambda s: 100 * s["pos_err_m"]["per_scenario_rms"]["median"], "{:.2f}"),
    ("", "per-scenario RMS: worst [cm]", lambda s: 100 * s["pos_err_m"]["per_scenario_rms"]["max"], "{:.2f}"),
    ("", "final error, last 0.5 s: mean [cm]", lambda s: 100 * s["final_pos_err_m"]["mean"], "{:.2f}"),
    ("", "final error, last 0.5 s: worst [cm]", lambda s: 100 * s["final_pos_err_m"]["max"], "{:.2f}"),
    ("Orientation error", None, None, None),
    ("", "RMS [deg]", lambda s: DEG * s["ori_err_rad"]["rms"], "{:.2f}"),
    ("", "mean [deg]", lambda s: DEG * s["ori_err_rad"]["mean"], "{:.2f}"),
    ("", "std [deg]", lambda s: DEG * s["ori_err_rad"]["std"], "{:.2f}"),
    ("", "95th percentile [deg]", lambda s: DEG * s["ori_err_rad"]["p95"], "{:.2f}"),
    ("", "max [deg]", lambda s: DEG * s["ori_err_rad"]["max"], "{:.2f}"),
    ("", "per-scenario RMS: mean [deg]", lambda s: DEG * s["ori_err_rad"]["per_scenario_rms"]["mean"], "{:.2f}"),
    ("", "per-scenario RMS: std [deg]", lambda s: DEG * s["ori_err_rad"]["per_scenario_rms"]["std"], "{:.2f}"),
    ("COM stability (rho: 0 = centred, 1 = at the footprint edge)", None, None, None),
    ("", "rho mean", lambda s: s["com_rho"]["mean"], "{:.3f}"),
    ("", "rho std", lambda s: s["com_rho"]["std"], "{:.3f}"),
    ("", "rho max", lambda s: s["com_rho"]["max"], "{:.3f}"),
    ("", "margin to edge, mean [cm]", lambda s: 100 * s["com_margin_m"]["mean"], "{:.2f}"),
    ("", "margin to edge, std [cm]", lambda s: 100 * s["com_margin_m"]["std"], "{:.2f}"),
    ("", "margin to edge, worst step [cm]", lambda s: 100 * s["com_margin_m"]["min"], "{:.2f}"),
    ("", "per-scenario closest margin: mean [cm]", lambda s: 100 * s["com_margin_m"]["per_scenario_min_mean"], "{:.2f}"),
    ("", "per-scenario closest margin: 5th pct [cm]", lambda s: 100 * s["com_margin_m"]["per_scenario_min_p5"], "{:.2f}"),
    ("Base motion", None, None, None),
    ("", "speed mean [m/s]", lambda s: s["base"]["speed_mean_mps"], "{:.3f}"),
    ("", "speed max [m/s]", lambda s: s["base"]["speed_max_mps"], "{:.3f}"),
    ("", "mean absolute yaw rate [rad/s]", lambda s: s["base"]["yaw_rate_abs_mean"], "{:.3f}"),
    ("", "max roll/pitch [deg]", lambda s: DEG * s["base"]["max_tilt_rad"], "{:.1f}"),
    ("", "max distance from spawn [m]", lambda s: s["base"]["max_distance_from_spawn_m"], "{:.2f}"),
    ("Solver", None, None, None),
    ("", "arm QP failures (n)", lambda s: s["ik_failures"], "{:d}"),
]


def load_runs(eval_dir: str | Path) -> dict[float, dict[str, dict]]:
    """``{payload_kg: {set: {logs..., "ik_failures": n}}}`` from the ``payload_<kg>kg`` folders of ``eval_dir``."""
    runs = {}
    for folder in sorted(Path(eval_dir).glob("payload_*kg")):
        payload = float(folder.name[len("payload_"):-len("kg")])
        summary = json.loads((folder / "summary.json").read_text())
        sets = {}
        for name in SET_NAMES:
            if (folder / f"{name}.npz").exists():
                data = np.load(folder / f"{name}.npz")
                part = {k: data[k] for k in (*LOG_FIELDS, "valid", "fail_step", "n_steps")}
                part["dt"] = float(data["dt"])
                part["ik_failures"] = summary["sets"][name]["ik_failures"]
                sets[name] = part
        runs[payload] = sets
    return dict(sorted(runs.items()))


def summarize_parts(parts: list[dict]) -> dict:
    """Summary statistics of the pooled scenarios of ``parts`` (each from ``load_runs``)."""
    pooled = pool_parts(parts)
    out = summarize({k: pooled[k] for k in LOG_FIELDS}, pooled["valid"], pooled["fail_step"], parts[0]["dt"],
                    n_steps=pooled["n_steps"])
    out["ik_failures"] = int(sum(p["ik_failures"] for p in parts))
    return out


def markdown_table(columns: dict[str, dict]) -> str:
    """Aligned pipe table: one row per metric, one column per entry of ``columns`` ({header: summary})."""
    header = ["metric", *columns]
    body = []
    for section, label, getter, fmt in ROWS:
        if label is None:
            body.append([f"**{section}**"] + [""] * len(columns))
            continue
        cells = []
        for summary in columns.values():
            value = getter(summary)
            cells.append("-" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value))
        body.append([label, *cells])
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    sep = "|" + "|".join(("-" * (w + 1) + ":") if i else ("-" * (w + 2)) for i, w in enumerate(widths)) + "|"
    return "\n".join([line(header), sep, *[line(r) for r in body]])


def _columns(parts_by_set: dict[str, list[dict]]) -> dict[str, dict]:
    columns = {"ALL": summarize_parts([p for parts in parts_by_set.values() for p in parts])}
    for name in SET_NAMES:
        if name in parts_by_set:
            columns[name] = summarize_parts(parts_by_set[name])
    return columns


def _kg(payload: float) -> str:
    return f"{payload:g} kg"


def build_report(eval_dir: str | Path) -> str:
    eval_dir = Path(eval_dir)
    runs = load_runs(eval_dir)
    if not runs:
        raise ValueError(f"no payload_<kg>kg folders in {eval_dir}")
    meta = json.loads((eval_dir / f"payload_{next(iter(runs)):g}kg" / "summary.json").read_text())["meta"]
    payloads = list(runs)
    sets = [n for n in SET_NAMES if all(n in r for r in runs.values())]
    n = {p: summarize_parts([runs[p][s] for s in sets])["scenarios"] for p in payloads}

    out = [
        f"# Evaluation report: {meta['name']}", "",
        f"- checkpoint: `{meta['checkpoint']}`, arm controller: `{meta['controller']}`, benchmark `{meta['benchmark_fingerprint']}` "
        f"(seed {meta['benchmark_seed']}), commit `{meta['git']}`, {meta['date']}",
        f"- extra mass at the gripper: {', '.join(_kg(p) for p in payloads)} (the same {meta['scenarios_per_set']} scenarios "
        f"per trajectory type each time; only the payload differs). Training used 0.8-3 kg, so 0 kg is outside the trained range.",
        f"- trajectory types: {', '.join(sets)}; {n[payloads[0]]} scenarios per payload, {sum(n.values())} in total",
        "- failure = the whole-robot COM leaves the platform footprint or the platform tips (roll/pitch > 28.6 deg); "
        "this is also the training termination criterion",
        "- every table: rows are metrics, columns are trajectory types, `ALL` pools all of them; definitions at the end", "",
    ]

    out += ["## 1. All payloads together (" + " + ".join(_kg(p) for p in payloads) + ")", "",
            markdown_table(_columns({s: [runs[p][s] for p in payloads] for s in sets})), "",
            "### 1b. By payload (all trajectory types pooled)", "",
            markdown_table({**{_kg(p): summarize_parts([runs[p][s] for s in sets]) for p in payloads},
                            "all": summarize_parts([runs[p][s] for p in payloads for s in sets])}), ""]
    for i, p in enumerate(payloads, start=2):
        out += [f"## {i}. Extra mass {_kg(p)}", "", markdown_table(_columns({s: [runs[p][s]] for s in sets})), ""]

    out += [
        "## Definitions", "",
        "- **time base**: the policy runs at 30 Hz; a metric only uses *valid* steps = steps inside the scenario's duration, "
        "up to and including the step where it failed",
        "- **failed / tipped / COM left footprint**: `failed` counts scenarios that failed within their duration. `tipped` and "
        "`COM left footprint` count scenarios where that flag was set at any time within the duration, also after the failure "
        "(the simulation keeps running), so `tipped` can exceed what ended a scenario",
        "- **position error**: distance between the TCP and the reference at the same time (env-local world frame); "
        "RMS, mean, std, percentiles and max pool all valid steps of all scenarios in the column (the std mixes the variation "
        "between scenarios and over time); `per-scenario RMS` first takes the RMS of each scenario, then the statistic across "
        "scenarios; `completed scenarios only` leaves out scenarios that failed; `final error` is the mean error over the "
        "last 0.5 s of each scenario that did not fail",
        "- **orientation error**: angle of the rotation between the TCP and reference orientation",
        "- **rho**: `max(|dx| / half_length, |dy| / half_width)` of the whole-robot COM relative to the footprint centre in the "
        "platform frame; 0 = centred, 1 = at the edge, above 1 = outside (only in failed scenarios)",
        "- **margin to edge**: `min(half_length - |dx|, half_width - |dy|)` in cm, negative = outside; depends on platform size, "
        "use rho to compare platforms",
        "- **base motion**: base-frame speed, yaw rate, roll/pitch and distance of the base from its spawn point",
        "- **arm QP failures**: solver failures of the arm controller during the runs", "",
    ]
    return "\n".join(out)
