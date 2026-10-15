import json

import numpy as np
import pytest

pytest.importorskip("torch")

from hybrid_control.rl.evaluation.metrics import LOG_FIELDS, failure_steps, pool_parts, summarize, valid_mask
from hybrid_control.rl.evaluation.report import ROWS, markdown_table, summarize_parts

DT = 1 / 30


def fake_part(seed, n=6, t=40, fail=(1, 4)):
    """A synthetic log of ``n`` scenarios of ``t`` steps; scenarios ``fail`` leave the footprint at step 10 and tip at 15."""
    rng = np.random.default_rng(seed)
    log = {k: rng.uniform(0.0, 1.0, (n, t, d)).astype(np.float32) for k, d in LOG_FIELDS.items()}
    log["tipped"][:] = 0
    log["left_footprint"][:] = 0
    for s in fail:
        log["left_footprint"][s, 10:] = 1
        log["tipped"][s, 15:] = 1
    n_steps = np.full(n, t)
    fail_step = failure_steps((log["tipped"][..., 0] > 0) | (log["left_footprint"][..., 0] > 0), n_steps)
    return {**log, "valid": valid_mask(n_steps, fail_step, t), "fail_step": fail_step, "n_steps": n_steps, "dt": DT,
            "ik_failures": 2}


def test_pooling_one_part_equals_its_own_summary():
    part = fake_part(0)
    expect = summarize({k: part[k] for k in LOG_FIELDS}, part["valid"], part["fail_step"], DT, n_steps=part["n_steps"])
    got = summarize_parts([part])
    assert json.dumps({k: v for k, v in got.items() if k != "ik_failures"}, sort_keys=True) == json.dumps(expect, sort_keys=True)


def test_pooling_pads_shorter_logs_and_adds_up_counts():
    a, b = fake_part(1, n=6, t=40), fake_part(2, n=4, t=25, fail=(0,))
    pooled = summarize_parts([a, b])
    assert pooled["scenarios"] == 10 and pooled["failed"] == 3 and pooled["ik_failures"] == 4
    assert pooled["tipped"] == 3 and pooled["left_footprint"] == 3
    # the pooled RMS is the RMS over all valid samples of both parts, not the mean of the two RMS values
    err = np.concatenate([p["pos_err"][..., 0][p["valid"]] for p in (a, b)])
    assert np.isclose(pooled["pos_err_m"]["rms"], np.sqrt(np.mean(err**2)), rtol=1e-6)
    assert np.isclose(pooled["pos_err_m"]["std"], err.std(), rtol=1e-6)
    assert np.isclose(pooled["failure_rate"], 0.3)


def test_failures_end_the_valid_steps_but_later_tipping_is_counted():
    part = fake_part(3)
    assert part["fail_step"][1] == 10 and part["valid"][1].sum() == 11  # up to and including the failing step
    s = summarize_parts([part])
    assert s["failed"] == 2 and s["tipped"] == 2


def test_table_has_a_row_per_metric_and_a_column_per_entry():
    summary = summarize_parts([fake_part(4)])
    table = markdown_table({"ALL": summary, "fixed": summary}).split("\n")
    assert len(table) == 2 + len(ROWS)
    assert table[0].count("|") == 4 and all(line.count("|") == 4 for line in table)
    assert "failure rate [%]" in "\n".join(table)
