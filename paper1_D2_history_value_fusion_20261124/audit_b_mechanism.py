"""Post-run audit of D2-B action activity and dynamic health-curve shape."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution

import b_run as runner

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "b_results_corrected"
RUN_MANIFEST = RUN_DIR / "B_RUN.json"
OUTPUT = RUN_DIR / "B_MECHANISM_AUDIT.json"
STRATEGY_PAIRS = (("RH_dynamic_minus_R1", 0, 3),
                  ("RH_dynamic_minus_RH_static", 1, 3),
                  ("RH_dynamic_minus_RH_scalar", 2, 3))


def clopper_pearson(successes: int, trials: int) -> list[float]:
    if not trials:
        return [0.0, 1.0]
    lower = float(beta_distribution.ppf(.025, successes, trials - successes + 1)) if successes else 0.0
    upper = float(beta_distribution.ppf(.975, successes + 1, trials - successes)) if successes < trials else 1.0
    return [lower, upper]


def rank_discordance(dynamic: np.ndarray, fixed: np.ndarray,
                     legal: np.ndarray) -> tuple[int, int]:
    indexes = np.triu(np.ones((len(runner.Q), len(runner.Q)), dtype=bool), 1)
    dynamic_difference = dynamic[:, :, None] - dynamic[:, None, :]
    fixed_difference = fixed[None, :, None] - fixed[None, None, :]
    dynamic_order = np.sign(dynamic_difference)
    fixed_order = np.sign(fixed_difference)
    eligible_pairs = legal[:, :, None] & legal[:, None, :] & indexes[None]
    non_tied = eligible_pairs & (dynamic_order != 0) & (fixed_order != 0)
    discordant = non_tied & (dynamic_order != fixed_order)
    return int(discordant.sum()), int(non_tied.sum())


def audit_scene(scene: int, spec: tuple, gains, record: dict) -> dict:
    model, graph, graph_path = runner.configure(spec, gains)
    fixed = runner.fixed_health_cost(model, runner.prepare_health_cost_affine(model))
    filename = RUN_DIR / f"scene_{scene:02}.npz"
    with np.load(filename, allow_pickle=False) as archive:
        fixture_results = {}
        all_alarm_events = 0
        all_action_differences = np.zeros(3, dtype=np.int64)
        all_legal_sizes = np.zeros(8, dtype=np.int64)
        all_q_counts = np.zeros((4, len(runner.Q)), dtype=np.int64)
        all_health_curve_discordant = 0
        all_health_curve_pairs = 0
        all_health_argmin_disagreements = 0
        fit_residuals = []
        scalar_denominators = 0
        scalar_zero_denominators = 0
        risk_margin_values = []

        for fixture in ("continuous", "burst"):
            masks = archive[f"{fixture}_masks"].astype(bool)
            scores = archive[f"{fixture}_scores"]
            actions = archive[f"{fixture}_actions"].astype(np.int64)
            dynamic = archive[f"{fixture}_dynamic_health_cost"][:, 3]
            target_alarm = masks[:, :, 6]
            runs = runner.alarm_runs(target_alarm)
            path_count = len(actions)
            history = np.full(path_count, 7 ** 6 - 1, dtype=np.int32)
            pair_events = 0
            pair_differences = np.zeros(3, dtype=np.int64)
            legal_sizes = np.zeros(8, dtype=np.int64)
            q_counts = np.zeros((4, len(runner.Q)), dtype=np.int64)
            curve_argmin_disagreement = 0
            curve_rank_discordant = 0
            curve_pairs = 0
            fixture_fit_residuals = []
            fixture_risk_margins = []
            scalar_zero = 0

            for tick in range(1, runner.T):
                legal = graph[history]
                in_window_alarm = target_alarm[:, tick] & (50 <= tick < 110)
                alarm_rows = np.flatnonzero(in_window_alarm)
                if alarm_rows.size:
                    alarm_legal = legal[alarm_rows]
                    alarm_dynamic = dynamic[alarm_rows, tick]
                    legal_sizes += np.bincount(alarm_legal.sum(axis=1), minlength=8)
                    for policy in range(4):
                        q_counts[policy] += np.bincount(
                            actions[alarm_rows, policy, tick], minlength=len(runner.Q)
                        )
                    pair_events += alarm_rows.size
                    for pair_index, (_, left, right) in enumerate(STRATEGY_PAIRS):
                        pair_differences[pair_index] += np.count_nonzero(
                            actions[alarm_rows, left, tick] != actions[alarm_rows, right, tick]
                        )

                    risk_cell = runner.cell_index(
                        scores[alarm_rows, tick, 6] / record["threshold"],
                        runs[alarm_rows, tick],
                        np.asarray(record["risk_table"]["boundaries"]),
                    )
                    mapped_cell = np.asarray(record["risk_table"]["fallback_cell"])[risk_cell]
                    risk = np.asarray(record["risk_table"]["simultaneous_upper"])[mapped_cell]
                    risk_action = np.asarray([
                        runner.select_q(value, legal_row)
                        for value, legal_row in zip(risk, alarm_legal)
                    ])
                    if not np.array_equal(risk_action, actions[alarm_rows, 0, tick]):
                        raise AssertionError("Archived R1 actions do not replay from the saved risk table.")
                    q_zero_legal = alarm_legal[:, 0]
                    has_nonzero = alarm_legal[:, 1:].any(axis=1)
                    comparable = q_zero_legal & has_nonzero
                    if comparable.any():
                        risk_margin = (
                            np.min(np.where(alarm_legal[:, 1:], risk[:, 1:], np.inf), axis=1)
                            - risk[:, 0]
                        )
                        fixture_risk_margins.extend(risk_margin[comparable].tolist())

                    dynamic_action = np.asarray([
                        runner.select_q(value, legal_row)
                        for value, legal_row in zip(alarm_dynamic, alarm_legal)
                    ])
                    fixed_action = np.asarray([
                        runner.select_q(fixed, legal_row)
                        for legal_row in alarm_legal
                    ])
                    curve_argmin_disagreement += int(
                        np.count_nonzero(dynamic_action != fixed_action)
                    )
                    discordant, comparable_pairs = rank_discordance(
                        alarm_dynamic, fixed, alarm_legal
                    )
                    curve_rank_discordant += discordant
                    curve_pairs += comparable_pairs

                    numerator = np.sum(alarm_dynamic * fixed[None, :] * alarm_legal, axis=1)
                    denominator = np.sum(fixed[None, :] ** 2 * alarm_legal, axis=1)
                    zero = denominator <= 1e-24
                    scalar_zero += int(zero.sum())
                    scalar = np.divide(
                        numerator, denominator, out=np.zeros_like(numerator), where=~zero
                    ).clip(0.0)
                    residual = np.linalg.norm(
                        (alarm_dynamic - scalar[:, None] * fixed[None, :]) * alarm_legal,
                        axis=1,
                    )
                    scale = np.linalg.norm(alarm_dynamic * alarm_legal, axis=1)
                    fixture_fit_residuals.extend(
                        np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 1e-12).tolist()
                    )

                history = (history * 7 + actions[:, 3, tick]) % (7 ** 6)

            fixture_results[fixture] = {
                "target_alarm_events": pair_events,
                "strategy_action_differences": {
                    name: {
                        "count": int(pair_differences[index]),
                        "rate": float(pair_differences[index] / pair_events) if pair_events else None,
                    }
                    for index, (name, _, _) in enumerate(STRATEGY_PAIRS)
                },
                "M7_legal_action_set_size_counts_on_alarm": legal_sizes.tolist(),
                "alarm_action_counts_by_policy_and_q": q_counts.tolist(),
                "risk_only_q0_legal_and_nonzero_risk_margin_event_count": len(fixture_risk_margins),
                "risk_only_nonzero_minus_q0_UCB_margin_mean": (
                    float(np.mean(fixture_risk_margins)) if fixture_risk_margins else None
                ),
                "dynamic_health_only_vs_static_health_only_argmin_disagreement_count": int(
                    curve_argmin_disagreement
                ),
                "dynamic_vs_static_pairwise_rank_discordance_count": int(curve_rank_discordant),
                "dynamic_vs_static_pairwise_rank_comparisons": int(curve_pairs),
                "dynamic_to_scalar_relative_fit_residual_median": (
                    float(np.median(fixture_fit_residuals)) if fixture_fit_residuals else None
                ),
                "dynamic_to_scalar_relative_fit_residual_p95": (
                    float(np.quantile(fixture_fit_residuals, .95)) if fixture_fit_residuals else None
                ),
                "scalar_zero_denominator_events": scalar_zero,
            }
            all_alarm_events += pair_events
            all_action_differences += pair_differences
            all_legal_sizes += legal_sizes
            all_q_counts += q_counts
            all_health_curve_discordant += curve_rank_discordant
            all_health_curve_pairs += curve_pairs
            all_health_argmin_disagreements += curve_argmin_disagreement
            fit_residuals.extend(fixture_fit_residuals)
            scalar_zero_denominators += scalar_zero
            risk_margin_values.extend(fixture_risk_margins)

    return {
        "scene": scene,
        "spec": list(spec),
        "M7_graph_sha256": runner.digest(graph_path),
        "attack_fixture_audit": fixture_results,
        "target_alarm_events": all_alarm_events,
        "strategy_action_differences": {
            name: {
                "count": int(all_action_differences[index]),
                "rate": float(all_action_differences[index] / all_alarm_events) if all_alarm_events else None,
            }
            for index, (name, _, _) in enumerate(STRATEGY_PAIRS)
        },
        "M7_legal_action_set_size_counts_on_alarm": all_legal_sizes.tolist(),
        "alarm_action_counts_by_policy_and_q": all_q_counts.tolist(),
        "dynamic_to_scalar_relative_fit_residual_median": (
            float(np.median(fit_residuals)) if fit_residuals else None
        ),
        "dynamic_to_scalar_relative_fit_residual_p95": (
            float(np.quantile(fit_residuals, .95)) if fit_residuals else None
        ),
        "dynamic_vs_static_pairwise_rank_discordance_rate": (
            float(all_health_curve_discordant / all_health_curve_pairs)
            if all_health_curve_pairs else None
        ),
        "dynamic_health_only_vs_static_health_only_argmin_disagreements": int(
            all_health_argmin_disagreements
        ),
        "dynamic_health_only_vs_static_health_only_argmin_disagreement_rate": (
            float(all_health_argmin_disagreements / all_alarm_events)
            if all_alarm_events else None
        ),
        "dynamic_vs_static_pairwise_rank_comparisons": int(all_health_curve_pairs),
        "scalar_zero_denominator_events": scalar_zero_denominators,
        "risk_only_nonzero_minus_q0_UCB_margin_mean": (
            float(np.mean(risk_margin_values)) if risk_margin_values else None
        ),
        "risk_only_nonzero_minus_q0_UCB_margin_min": (
            float(np.min(risk_margin_values)) if risk_margin_values else None
        ),
    }


def verify_q1_regression(gains, manifest: dict) -> dict:
    maximum_full_reference_error = 0.0
    maximum_literal_update_error = 0.0
    maximum_frozen_b1_error = 0.0
    namespace = runner.SMOKE_NAMESPACE
    for scene, spec in enumerate(runner.health.scenarios()):
        model, _, _ = runner.configure(spec, gains)
        paths = runner.health.physical(
            model, 2, runner.make_seed(scene, 0, 0, 11, 0, namespace)
        )
        scores = runner.h1_score(model, paths)
        threshold = manifest["results"][str(scene)]["threshold"]
        masks = scores > threshold
        zero_delta = np.zeros((2, runner.T, 7, 2))
        optimized = runner.simulate_full_accept_reference(
            model, paths, scores, masks, zero_delta
        )
        independent = runner.full_accept_reference(
            model, paths, masks, zero_delta
        )
        np.testing.assert_allclose(optimized, independent, rtol=2e-9, atol=2e-8)
        maximum_full_reference_error = max(
            maximum_full_reference_error,
            float(np.max(np.abs(optimized - independent))),
        )

        q1_actions = np.full((2, len(runner.STRATEGIES), runner.T), 6, dtype=np.int8)
        literal = runner.literal_policy_replay(model, paths, zero_delta, masks, q1_actions)
        np.testing.assert_allclose(literal[:, 0], optimized, rtol=2e-9, atol=2e-8)
        maximum_literal_update_error = max(
            maximum_literal_update_error,
            float(np.max(np.abs(literal[:, 0] - optimized))),
        )

        packed_masks = np.sum(masks * runner.BITS[None, None, :], axis=2).astype(np.uint8)
        frozen_b1 = runner.health.evaluate(model, paths, packed_masks)
        np.testing.assert_allclose(
            optimized[:, 2], frozen_b1["missing"][:, 6], rtol=2e-9, atol=2e-8
        )
        np.testing.assert_allclose(
            optimized[:, 4], frozen_b1["network"][:, 6], rtol=2e-9, atol=2e-8
        )
        maximum_frozen_b1_error = max(
            maximum_frozen_b1_error,
            float(np.max(np.abs(optimized[:, 2] - frozen_b1["missing"][:, 6]))),
            float(np.max(np.abs(optimized[:, 4] - frozen_b1["network"][:, 6]))),
        )

    return {
        "status": "PASS",
        "scene_count": len(runner.health.scenarios()),
        "smoke_paths_per_scene": 2,
        "seed_namespace": list(namespace),
        "formal_D2B_experiment_paths_generated": False,
        "max_full_accept_implementation_difference": maximum_full_reference_error,
        "max_literal_vs_optimized_update_difference": maximum_literal_update_error,
        "max_frozen_B1_regression_difference": maximum_frozen_b1_error,
    }


def main() -> None:
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_D2_B_development_screening":
        raise ValueError("Mechanism audit requires the complete corrected D2-B run.")
    if manifest.get("execution_commit") != "ac4d58c4ccac69fb101ab40baa7d9a06e3bffdf2":
        raise ValueError("Unexpected corrected D2-B run commit.")
    results = {}
    with np.load(runner.modes.OUTPUT / "mode_tables.npz", allow_pickle=False) as gains:
        q1_regression = verify_q1_regression(gains, manifest)
        for scene, spec in enumerate(runner.health.scenarios()):
            results[str(scene)] = audit_scene(scene, spec, gains, manifest["results"][str(scene)])
    totals = sum(row["target_alarm_events"] for row in results.values())
    actions = {name: sum(row["strategy_action_differences"][name]["count"]
                         for row in results.values())
               for name, _, _ in STRATEGY_PAIRS}
    summary = {
        "status": "posthoc_D2_B_mechanism_audit",
        "execution_commit": manifest["execution_commit"],
        "protocol_commit": manifest["protocol_commit"],
        "B_RUN_sha256": runner.digest(RUN_MANIFEST),
        "mechanism_audit_source_sha256": runner.digest(Path(__file__)),
        "claim_scope": "descriptive_development_screening_only",
        "q1_deterministic_regression": q1_regression,
        "target_alarm_events_in_attack_windows": totals,
        "strategy_action_differences": {
            name: {"count": count, "rate": count / totals if totals else None}
            for name, count in actions.items()
        },
        "alarm_action_counts_by_policy_and_q": np.sum(
            [row["alarm_action_counts_by_policy_and_q"] for row in results.values()], axis=0
        ).tolist(),
        "M7_legal_action_set_size_counts_on_alarm": np.sum(
            [row["M7_legal_action_set_size_counts_on_alarm"] for row in results.values()], axis=0
        ).tolist(),
        "dynamic_to_scalar_relative_fit_residual_median_equal_scene": float(np.mean([
            row["dynamic_to_scalar_relative_fit_residual_median"] for row in results.values()
        ])),
        "dynamic_to_scalar_relative_fit_residual_p95_max_scene": float(np.max([
            row["dynamic_to_scalar_relative_fit_residual_p95"] for row in results.values()
        ])),
        "dynamic_vs_static_pairwise_rank_discordance_rate_equal_scene": float(np.mean([
            row["dynamic_vs_static_pairwise_rank_discordance_rate"] for row in results.values()
        ])),
        "dynamic_vs_static_health_only_argmin_disagreement_rate_equal_scene": float(np.mean([
            row["dynamic_health_only_vs_static_health_only_argmin_disagreement_rate"]
            for row in results.values()
        ])),
        "risk_only_nonzero_minus_q0_UCB_margin_mean_equal_scene": float(np.mean([
            row["risk_only_nonzero_minus_q0_UCB_margin_mean"] for row in results.values()
        ])),
        "risk_only_nonzero_minus_q0_UCB_margin_min_scene": float(np.min([
            row["risk_only_nonzero_minus_q0_UCB_margin_min"] for row in results.values()
        ])),
        "scenes": results,
    }
    OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "status": summary["status"],
        "target_alarm_events": totals,
        "strategy_action_differences": summary["strategy_action_differences"],
        "M7_legal_action_set_size_counts": summary["M7_legal_action_set_size_counts_on_alarm"],
        "dynamic_vs_scalar_residual_median": summary["dynamic_to_scalar_relative_fit_residual_median_equal_scene"],
        "dynamic_vs_static_rank_discordance": summary["dynamic_vs_static_pairwise_rank_discordance_rate_equal_scene"],
        "dynamic_vs_static_health_only_argmin_disagreement_rate_equal_scene": float(np.mean([
            row["dynamic_health_only_vs_static_health_only_argmin_disagreement_rate"]
            for row in results.values()
        ])),
        "risk_nonzero_minus_q0_margin_mean": summary["risk_only_nonzero_minus_q0_UCB_margin_mean_equal_scene"],
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
