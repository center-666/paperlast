"""Independent replay audit for completed D2-B2 archives.

This verifier reimplements the discrete action constraints and risk argmin in
this file instead of importing the D2-B2 runner's controller functions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "b2_results"
RUN_MANIFEST = RUN_DIR / "B2_RUN.json"
ANALYSIS_FILE = RUN_DIR / "B2_ANALYSIS.json"
OUTPUT = RUN_DIR / "B2_INDEPENDENT_AUDIT.json"
Q = np.asarray([0., .2, .4, .6, .8, .9, 1.])
EDGE_BITS = 1 << np.arange(7)
TIME = 150
START, STOP = 50, 110
M7_STATE_COUNT = 7 ** 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alarm_run_lengths(target_alarm: np.ndarray) -> np.ndarray:
    runs = np.zeros(target_alarm.shape, dtype=np.int32)
    for tick in range(1, target_alarm.shape[1]):
        runs[:, tick] = np.where(target_alarm[:, tick], runs[:, tick - 1] + 1, 0)
    return runs


def score_cells(scores, runs, threshold, boundaries):
    normalized = scores / threshold
    score_bin = np.searchsorted(boundaries, normalized, side="right")
    duration_bin = np.where(runs == 1, 0, np.where(runs <= 3, 1, 2))
    return 3 * score_bin + duration_bin


def larger_q_argmin(cost, feasible):
    eligible = np.where(feasible, cost, np.inf)
    best = eligible.min(axis=1)
    tied = feasible & (eligible <= best[:, None] + 1e-10)
    if not np.all(tied.any(axis=1)):
        raise AssertionError("The independently reconstructed feasible set is empty.")
    return (len(Q) - 1) - np.argmax(tied[:, ::-1], axis=1)


def replay_scene_fixture(scene, fixture, result, archive, graph):
    actions = archive[f"{fixture}_actions"].astype(np.int64)
    masks = archive[f"{fixture}_masks"].astype(bool)
    scores = archive[f"{fixture}_scores"]
    saved_sets = archive[f"{fixture}_feasible_sets"].astype(bool)
    relative = archive[f"{fixture}_relative_health_costs"]
    absolute = archive[f"{fixture}_absolute_health_costs"]
    scalar = archive[f"{fixture}_scalar_fit"]
    static_absolute = np.asarray(result["static_nominal_health_cost_absolute"])
    static_relative = np.asarray(result["static_nominal_health_cost_relative_to_q1"])
    epsilon = float(result["epsilon"])
    n = len(actions)
    history = np.full((n, 4), M7_STATE_COUNT - 1, dtype=np.int32)
    runs = alarm_run_lengths(masks[:, :, 6])
    risk_table = result["risk_table"]
    boundaries = np.asarray(risk_table["boundaries"])
    fallback = np.asarray(risk_table["fallback_cell"])
    upper = np.asarray(risk_table["simultaneous_upper"])
    action_replay_errors = 0
    m7_violations = 0
    budget_violations = 0
    saved_feasible_mismatches = 0
    alarm_decisions = 0
    q_counts = np.zeros((4, len(Q)), dtype=np.int64)
    feasible_sizes = np.zeros(4, dtype=np.int64)
    q0_feasible = np.zeros(4, dtype=np.int64)
    static_dynamic_set_mismatch = 0
    static_scalar_set_mismatch = 0

    for tick in range(1, TIME):
        legal = graph[history]
        chosen = actions[:, :, tick]
        m7_violations += int((~legal[np.arange(n)[:, None], np.arange(4)[None], chosen]).sum())
        alarm = masks[:, tick, 6]
        alarm_rows = np.flatnonzero(alarm)
        if alarm_rows.size:
            alarm_decisions += int(alarm_rows.size)
            cell = score_cells(
                scores[alarm_rows, tick, 6] / float(result["threshold"]),
                runs[alarm_rows, tick],
                1.0,
                boundaries,
            )
            mapped = fallback[cell]
            attack_cost = upper[mapped]
            for policy in range(4):
                policy_legal = legal[alarm_rows, policy]
                if policy == 0:
                    policy_feasible = policy_legal
                elif policy == 1:
                    policy_feasible = policy_legal & (
                        static_relative[None, :]
                        <= epsilon * static_absolute[-1] + 1e-10
                    )
                elif policy == 2:
                    policy_feasible = policy_legal & (
                        scalar[alarm_rows, policy, tick, None] * static_relative[None, :]
                        <= epsilon * absolute[alarm_rows, policy, tick, -1, None] + 1e-10
                    )
                else:
                    policy_feasible = policy_legal & (
                        relative[alarm_rows, policy, tick]
                        <= epsilon * absolute[alarm_rows, policy, tick, -1, None] + 1e-10
                    )
                saved = saved_sets[alarm_rows, policy, tick]
                saved_feasible_mismatches += int(np.count_nonzero(saved != policy_feasible))
                if policy == 1:
                    static_dynamic_set_mismatch += int(np.count_nonzero(
                        saved_sets[alarm_rows, 1, tick] != saved_sets[alarm_rows, 3, tick]
                    ))
                    static_scalar_set_mismatch += int(np.count_nonzero(
                        saved_sets[alarm_rows, 1, tick] != saved_sets[alarm_rows, 2, tick]
                    ))
                feasible_sizes[policy] += int(policy_feasible.sum())
                q0_feasible[policy] += int(policy_feasible[:, 0].sum())
                expected = larger_q_argmin(attack_cost, policy_feasible)
                selected_actions = chosen[alarm_rows, policy]
                action_replay_errors += int(np.count_nonzero(expected != selected_actions))
                q_counts[policy] += np.bincount(selected_actions, minlength=len(Q))

                if policy == 1:
                    selected_health = static_relative[selected_actions]
                    baseline = static_absolute[-1]
                elif policy == 2:
                    selected_health = (
                        scalar[alarm_rows, policy, tick]
                        * static_relative[selected_actions]
                    )
                    baseline = absolute[alarm_rows, policy, tick, -1]
                elif policy == 3:
                    selected_health = relative[
                        alarm_rows, policy, tick, selected_actions
                    ]
                    baseline = absolute[alarm_rows, policy, tick, -1]
                else:
                    selected_health = np.zeros(len(alarm_rows))
                    baseline = np.ones(len(alarm_rows))
                if policy:
                    budget_violations += int(np.count_nonzero(
                        selected_health > epsilon * baseline + 1e-10
                    ))
        nonalarm_rows = np.flatnonzero(~alarm)
        if nonalarm_rows.size:
            action_replay_errors += int(
                np.count_nonzero(chosen[nonalarm_rows] != len(Q) - 1)
            )
        history = (history * 7 + chosen) % M7_STATE_COUNT

    if saved_feasible_mismatches or action_replay_errors or m7_violations or budget_violations:
        raise AssertionError(
            "Independent constrained-policy replay did not match archived decisions."
        )
    return {
        "alarm_decisions": alarm_decisions,
        "M7_violations": m7_violations,
        "health_budget_violations": budget_violations,
        "saved_feasible_set_mismatches": saved_feasible_mismatches,
        "risk_argmin_or_off_alarm_action_mismatches": action_replay_errors,
        "alarm_action_counts_by_policy_and_q": q_counts.tolist(),
        "M7_feasible_action_counts_on_alarm_by_policy": feasible_sizes.tolist(),
        "q0_feasible_counts_on_alarm_by_policy": q0_feasible.tolist(),
        "HC_static_HC_dynamic_feasible_set_mismatches": static_dynamic_set_mismatch,
        "HC_static_HC_scalar_feasible_set_mismatches": static_scalar_set_mismatch,
    }


def main():
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_D2_B2_development_screening":
        raise ValueError("Independent audit requires a complete D2-B2 run.")
    if manifest.get("execution_commit") != "d1cfa56095f7e57d9b95356f930c4eb4123c445b":
        raise ValueError("Unexpected D2-B2 runner commit.")
    policy_names = manifest["strategies"]
    if policy_names != ["R1", "HC_static", "HC_scalar", "HC_dynamic"]:
        raise ValueError("D2-B2 policy order changed.")
    if manifest["seed_namespace"] != [20261125, 7319, 1]:
        raise ValueError("D2-B2 seed namespace changed.")
    scene_reports = {}
    policy_attack_difference = np.zeros((len(policy_names), 5))
    policy_health_difference = np.zeros((len(policy_names), 5))
    q1_health_difference = np.zeros(5)
    total_m7 = total_budget = total_feasible_mismatch = total_decision_mismatch = 0
    total_alarm = 0
    static_dynamic_set_mismatch = 0
    static_scalar_set_mismatch = 0
    independent_q_counts = np.zeros((len(policy_names), len(Q)), dtype=np.int64)
    independent_feasible_counts = np.zeros(len(policy_names), dtype=np.int64)
    independent_q0_counts = np.zeros(len(policy_names), dtype=np.int64)
    for scene in range(12):
            record = manifest["results"][str(scene)]
            dynamics, topology, sender, _ = record["spec"]
            graph_file = HERE / "experiment_A" / "graphs" / f"{topology}_{dynamics}_target{sender}_qwords.npz"
            with np.load(graph_file, allow_pickle=False) as graph_archive:
                graph = graph_archive["legal"].copy()
            archive_path = RUN_DIR / f"scene_{scene:02}.npz"
            expected_hash = manifest["output_sha256"][archive_path.name]
            if sha256(archive_path) != expected_hash:
                raise ValueError(f"Scene archive hash mismatch: {archive_path.name}")
            scene_policy_loss = np.zeros((len(policy_names), 5))
            scene_q1_loss = np.zeros(5)
            fixture_audits = {}
            with np.load(archive_path, allow_pickle=False) as archive:
                for fixture in ("continuous", "burst"):
                    replay = replay_scene_fixture(scene, fixture, record, archive, graph)
                    fixture_audits[fixture] = replay
                    total_m7 += replay["M7_violations"]
                    total_budget += replay["health_budget_violations"]
                    total_feasible_mismatch += replay["saved_feasible_set_mismatches"]
                    total_decision_mismatch += replay["risk_argmin_or_off_alarm_action_mismatches"]
                    total_alarm += replay["alarm_decisions"]
                    static_dynamic_set_mismatch += replay[
                        "HC_static_HC_dynamic_feasible_set_mismatches"
                    ]
                    static_scalar_set_mismatch += replay[
                        "HC_static_HC_scalar_feasible_set_mismatches"
                    ]
                    independent_q_counts += np.asarray(
                        replay["alarm_action_counts_by_policy_and_q"]
                    )
                    independent_feasible_counts += np.asarray(
                        replay["M7_feasible_action_counts_on_alarm_by_policy"]
                    )
                    independent_q0_counts += np.asarray(
                        replay["q0_feasible_counts_on_alarm_by_policy"]
                    )
                    losses = archive[f"{fixture}_loss"]
                    reference = archive[f"{fixture}_full_accept_loss"]
                    scene_policy_loss += losses.mean(axis=0) / 2.
                    scene_q1_loss += reference.mean(axis=0) / 2.
                    np.testing.assert_allclose(
                        losses.mean(axis=0),
                        np.asarray(record["attacks"][fixture]["loss_mean"]),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                health_loss = archive["health_loss"].mean(axis=0)
                health_q1 = archive["health_full_accept_loss"].mean(axis=0)
                np.testing.assert_allclose(health_loss, record["health_loss_mean"],
                                           rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(health_q1, record["health_q1_reference_mean"],
                                           rtol=1e-12, atol=1e-12)
            policy_attack_difference += scene_policy_loss / 12.
            policy_health_difference += health_loss / 12.
            q1_health_difference += health_q1 / 12.
            scene_reports[str(scene)] = {
                "spec": record["spec"],
                "epsilon": record["epsilon"],
                "fixture_replay": fixture_audits,
                "attack_loss_mean": scene_policy_loss.tolist(),
                "health_loss_mean": health_loss.tolist(),
                "q1_health_mean": health_q1.tolist(),
            }

    analysis_attack = np.asarray([
        analysis["attack"]["policy_mean_mse"][policy]
        for policy in policy_names
    ])
    np.testing.assert_allclose(policy_attack_difference, analysis_attack,
                               rtol=1e-11, atol=1e-11)
    analysis_health = np.asarray([
        [analysis["health"]["mean_mse"][policy]["missing_post20"],
         analysis["health"]["mean_mse"][policy]["network_post20"]]
        for policy in policy_names
    ])
    np.testing.assert_allclose(
        policy_health_difference[:, [2, 4]], analysis_health,
        rtol=1e-11, atol=1e-11,
    )
    np.testing.assert_allclose(
        q1_health_difference[[2, 4]],
        [analysis["health"]["mean_mse"]["full_accept"]["missing_post20"],
         analysis["health"]["mean_mse"]["full_accept"]["network_post20"]],
        rtol=1e-11, atol=1e-11,
    )
    report = {
        "status": "PASS_independent_D2_B2_action_and_output_replay",
        "execution_commit": manifest["execution_commit"],
        "protocol_commit": manifest["protocol_commit"],
        "B2_RUN_sha256": sha256(RUN_MANIFEST),
        "B2_ANALYSIS_sha256": sha256(ANALYSIS_FILE),
        "audit_source_sha256": sha256(Path(__file__)),
        "independent_method": "reimplement alarm runs, M7 history, static/scalar/dynamic budget sets, and larger-q risk argmin without importing the B2 runner",
        "target_alarm_decisions_replayed": total_alarm,
        "M7_violations": total_m7,
        "health_budget_violations": total_budget,
        "saved_feasible_set_mismatches": total_feasible_mismatch,
        "risk_argmin_or_off_alarm_action_mismatches": total_decision_mismatch,
        "HC_static_HC_dynamic_feasible_set_mismatches": static_dynamic_set_mismatch,
        "HC_static_HC_scalar_feasible_set_mismatches": static_scalar_set_mismatch,
        "independent_alarm_action_counts_by_policy_and_q": independent_q_counts.tolist(),
        "independent_M7_feasible_action_counts_on_alarm_by_policy": independent_feasible_counts.tolist(),
        "independent_q0_feasible_counts_on_alarm_by_policy": independent_q0_counts.tolist(),
        "path_loss_weighted_summary": policy_attack_difference.tolist(),
        "analysis_policy_summary": analysis_attack.tolist(),
        "path_weighted_healthy_summary": policy_health_difference.tolist(),
        "full_accept_healthy_summary": q1_health_difference.tolist(),
        "scenes": scene_reports,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in (
        "status", "target_alarm_decisions_replayed", "M7_violations",
        "health_budget_violations", "saved_feasible_set_mismatches",
        "risk_argmin_or_off_alarm_action_mismatches"
    )}, indent=2))


if __name__ == "__main__":
    main()
