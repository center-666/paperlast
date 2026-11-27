"""Independent discrete replay and literal estimator audit for task-3 development."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import models_quality as quality
import task3

HERE = Path(__file__).resolve().parent
ROOT = HERE / "task3_dev"
POLICIES = task3.POLICIES


def choose_risk(table, score, run, threshold):
    if table["mode"] == "H_pooled":
        cell = 0
    else:
        duration = 0 if run == 1 else (1 if run <= 3 else 2)
        if table["mode"] == "D0_duration":
            cell = duration
        else:
            score_bin = int(np.searchsorted(np.asarray(table["score_tertiles"]),
                                            score/threshold, side="right"))
            cell = 3*score_bin+duration
    return np.asarray(table["means"])[cell]


def independent_actions(graph, masks, threshold, matched, saved_runs, risk_tables,
                        quality_model, health_masks):
    n = len(matched)
    alarm = masks[:, :, 6].astype(bool)
    histories = np.full((n, 8), 7**6-1, dtype=np.int64)
    actions = np.full((n, 8, 150), 6, dtype=np.int8)
    m7_violation = np.zeros(8, dtype=np.int64)
    budget_violation = np.zeros(8, dtype=np.int64)
    fixed_q = float(quality_model["fixed_T_q"])
    fixed_index = int(np.argmin(abs(quality.Q-fixed_q)))
    for tick in range(1, 150):
        legal = graph[histories]
        chosen = np.full((n, 8), 6, dtype=np.int8)
        on = np.flatnonzero(alarm[:, tick])
        chosen[on, 0] = 0
        chosen[on, 1] = np.argmax(legal[on, 1], axis=1)
        for path in on:
            allowed = np.flatnonzero(legal[path, 2] & (np.arange(7) >= fixed_index))
            chosen[path, 2] = allowed[0]
            for policy, constraint, table_index in (
                (3, 1, 0), (4, 2, 0), (5, 2, 1), (6, 2, 2)):
                feasible = legal[path, policy] & health_masks[constraint,
                    histories[path, policy]]
                if not feasible.any():
                    feasible = legal[path, policy].copy()
                    feasible[6] = True
                    if not feasible.any():
                        raise AssertionError("q1 fallback is absent from the M7 graph.")
                risk = choose_risk(risk_tables[table_index], matched[path, tick, 6],
                                   saved_runs[path, tick], threshold)
                objective = np.where(feasible, risk, np.inf)
                best = objective.min()
                chosen[path, policy] = np.flatnonzero(
                    feasible & (objective <= best+1e-10))[-1]
            chosen[path, 7] = 6
        invalid = ~legal[np.arange(n)[:, None], np.arange(8)[None], chosen]
        m7_violation += invalid.sum(axis=0)
        # A raw hard rejection may fail M7, and this is deliberately counted.
        if np.any(invalid[:, 1:]):
            raise AssertionError("An M7-projected development policy violated its graph.")
        for policy, constraint in ((3, 1), (4, 2), (5, 2), (6, 2)):
            if len(on):
                feasible_health = health_masks[constraint, histories[on, policy]]
                budget_violation[policy] += int(np.count_nonzero(
                    ~feasible_health[np.arange(len(on)), chosen[on, policy]]))
        actions[:, :, tick] = chosen
        histories = (histories*7+chosen) % 7**6
    return actions, m7_violation, budget_violation


def audit():
    run_path = ROOT / "RUN.json"
    run = quality.json.loads(run_path.read_text(encoding="utf-8"))
    if run["status"] != "complete_task3_development":
        raise ValueError("Task-3 development archive is incomplete.")
    for filename, expected in run["archives"].items():
        if task3.digest(ROOT / filename) != expected:
            raise ValueError(f"Task-3 archive hash mismatch: {filename}")
    total_actions = 0
    m7 = np.zeros(8, dtype=np.int64)
    budgets = np.zeros(8, dtype=np.int64)
    literal_replays = 0
    maximum_literal_error = 0.
    first_mismatch = None
    m7_legal_action_counts = {name: np.zeros((8, 7), dtype=np.int64)
                              for name in task3.QUALITY_ORDER}
    selected_action_counts = {name: np.zeros((8, 7), dtype=np.int64)
                              for name in task3.QUALITY_ORDER}
    with np.load(quality.frozen.modes.OUTPUT / "mode_tables.npz") as gains:
        for index, name in enumerate(task3.QUALITY_ORDER):
            config_record = run["configs"][name]
            model = task3.audit_frontend.load_model(index)
            quality_record = task3.read_quality_model(index)
            graph_path = HERE / "quality_models" / name / \
                f"{quality_record['topology']}_{quality_record['dynamics']}_target35_qwords.npz"
            with np.load(graph_path) as saved:
                graph = saved["legal"].copy()
            with np.load(HERE / "quality_models" / f"{name}.npz") as saved:
                health_masks = saved["policy_masks"].copy()
            with np.load(ROOT / f"config{index:02}.npz") as archive:
                for fixture in ("continuous", "burst", "health"):
                    prefix = fixture+"_"
                    data_masks = archive[prefix+"gates"].astype(bool)
                    matches = archive[prefix+"matched"]
                    saved_histories = archive[prefix+"histories"].astype(np.int64)
                    runs = np.zeros((len(data_masks), 150), dtype=np.int16)
                    for tick in range(1, 150):
                        runs[:, tick] = np.where(data_masks[:, tick, 6],
                                                 runs[:, tick-1]+1, 0)
                    tables = config_record["risk_and_loss"]["continuous"]["risk_tables"]
                    expected_actions, violation, budget_violation = independent_actions(
                        graph, data_masks, config_record["thresholds"]["no_repeat_matched"],
                        matches, runs, tables, quality_record, health_masks)
                    actual = archive[prefix+"actions"].astype(np.int8)
                    alarm_mask = data_masks[:, :, 6].astype(bool)
                    for policy in range(8):
                        legal_now = graph[saved_histories[:, policy]]
                        for q_index in range(7):
                            m7_legal_action_counts[name][policy, q_index] += int(
                                np.count_nonzero(alarm_mask & legal_now[:, :, q_index]))
                            selected_action_counts[name][policy, q_index] += int(
                                np.count_nonzero(alarm_mask & (actual[:, policy] == q_index)))
                    mismatch = np.argwhere(expected_actions != actual)
                    if len(mismatch) and first_mismatch is None:
                        path, policy, tick = map(int, mismatch[0])
                        first_mismatch = {"configuration": name, "fixture": fixture,
                                          "path": path, "policy": POLICIES[policy],
                                          "tick": tick,
                                          "expected": int(expected_actions[path, policy, tick]),
                                          "saved": int(actual[path, policy, tick])}
                    np.testing.assert_array_equal(expected_actions, actual)
                    m7 += violation
                    budgets += budget_violation
                    total_actions += actual.size
                    if fixture != "health":
                        path_obj = SimpleNamespace(
                            x=archive[prefix+"x"][:16], y=archive[prefix+"y"][:16])
                        delta = archive[prefix+"delta"][:16]
                        small_masks = data_masks[:16]
                        small_actions = actual[:16, :4]
                        literal = quality.frozen.common.literal_policy_replay(
                            model, path_obj, delta, small_masks, small_actions)
                        saved_loss = archive[prefix+"losses"][:16, :4]
                        # Frozen literal columns 0..4 map to task-3 columns 0,4,5,3,6.
                        for old_column, new_column in enumerate((0, 4, 5, 3, 6)):
                            error = float(np.max(abs(
                                literal[:, :, old_column]
                                - saved_loss[:, :, new_column])))
                            maximum_literal_error = max(maximum_literal_error, error)
                        np.testing.assert_allclose(
                            literal, saved_loss[:, :, (0, 4, 5, 3, 6)],
                            rtol=2e-9, atol=2e-8)
                        literal_replays += 16
    report = {
        "status": "PASS", "source_sha256": task3.digest(Path(__file__)),
        "run_sha256": task3.digest(run_path), "path_decision_entries_recomputed": total_actions,
        "first_action_mismatch": first_mismatch, "M7_violations_by_policy": m7.tolist(),
        "budget_violations_by_policy": budgets.tolist(),
        "literal_per_neighbor_replay_paths": literal_replays,
        "literal_maximum_absolute_error": maximum_literal_error,
        "m7_legal_alarm_opportunities_by_configuration_policy_q": {
            name: values.tolist() for name, values in m7_legal_action_counts.items()},
        "selected_alarm_actions_by_configuration_policy_q": {
            name: values.tolist() for name, values in selected_action_counts.items()},
    }
    quality.frozen.atomic_json(ROOT / "VERIFY.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    audit()
