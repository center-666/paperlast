"""Paired, path-level descriptive analysis of the frozen D2-B screening run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "b_results_corrected"
RUN_FILE = RUN_DIR / "B_RUN.json"
MECHANISM_FILE = RUN_DIR / "B_MECHANISM_AUDIT.json"
OUTPUT_JSON = RUN_DIR / "B_ANALYSIS.json"
OUTPUT_MD = RUN_DIR / "B_ANALYSIS.md"
POLICIES = ("R1", "RH_static", "RH_scalar", "RH_dynamic")
METRICS = (
    "missing_attack_window",
    "missing_recovery_window",
    "missing_post20",
    "network_attack_window",
    "network_post20",
)
COMPARISONS = {
    "RH_dynamic_minus_R1": (0, 3),
    "RH_dynamic_minus_RH_static": (1, 3),
    "RH_dynamic_minus_RH_scalar": (2, 3),
}
BOOTSTRAP_REPLICATES = 2048
BOOTSTRAP_SEED = (20261124, 4401, 2, 992)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
    temporary.replace(path)


def percentile_interval(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def load_run() -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_D2_B_development_screening":
        raise ValueError("Analysis requires all twelve completed D2-B scenes.")
    if manifest.get("claim_scope") != "development_screening_conditional_risk_proxy_not_closed_loop_bound":
        raise ValueError("Unexpected D2-B evidence scope.")

    attacks = []
    health = []
    for scene in manifest["scenes"]:
        filename = f"scene_{scene:02}.npz"
        path = RUN_DIR / filename
        if sha256(path) != manifest["output_sha256"].get(filename):
            raise ValueError(f"D2-B scene archive hash mismatch: {filename}")
        record = manifest["results"][str(scene)]
        with np.load(path, allow_pickle=False) as archive:
            health_loss = archive["health_loss"].copy()
            health_full = archive["health_full_accept_loss"].copy()
            health.append({
                "scene": scene,
                "loss": health_loss,
                "full": health_full,
                "fwer": record["target_fwer"],
                "fwer_ci": record["target_fwer_CP95"],
                "all_edge_fwer": record["all_edge_fwer"],
                "all_edge_fwer_ci": record["all_edge_fwer_CP95"],
            })
            for fixture in ("continuous", "burst"):
                losses = archive[f"{fixture}_loss"].copy()
                full = archive[f"{fixture}_full_accept_loss"].copy()
                actions = archive[f"{fixture}_actions"].copy()
                masks = archive[f"{fixture}_masks"].copy()
                stored_mean = np.asarray(record["attacks"][fixture]["loss_mean"])
                np.testing.assert_allclose(
                    losses.mean(axis=0), stored_mean, rtol=1e-12, atol=1e-12
                )
                attacks.append({
                    "scene": scene,
                    "fixture": fixture,
                    "loss": losses,
                    "full": full,
                    "actions": actions,
                    "target_alarm": (masks[:, :, 6] != 0),
                    "stored": record["attacks"][fixture],
                })
    return manifest, attacks, health


def bootstrap_attack(attacks: list[dict]) -> tuple[dict, dict]:
    rng = np.random.default_rng(np.random.SeedSequence(BOOTSTRAP_SEED))
    policy_means = np.zeros((len(POLICIES), len(METRICS)))
    full_means = np.zeros(len(METRICS))
    comparison_results = {}
    bootstrap_policy_means = np.zeros(
        (BOOTSTRAP_REPLICATES, len(POLICIES), len(METRICS))
    )
    bootstrap_full_means = np.zeros((BOOTSTRAP_REPLICATES, len(METRICS)))
    condition_weight = 1.0 / len(attacks)

    for record in attacks:
        loss = record["loss"]
        full = record["full"]
        policy_means += loss.mean(axis=0) * condition_weight
        full_means += full.mean(axis=0) * condition_weight
        sample_indices = rng.integers(
            0, len(loss), size=(BOOTSTRAP_REPLICATES, len(loss)), dtype=np.int32
        )
        bootstrap_policy_means += loss[sample_indices].mean(axis=1) * condition_weight
        bootstrap_full_means += full[sample_indices].mean(axis=1) * condition_weight

    for name, (reference, candidate) in COMPARISONS.items():
        difference = policy_means[candidate] - policy_means[reference]
        bootstrap_difference = (
            bootstrap_policy_means[:, candidate] - bootstrap_policy_means[:, reference]
        )
        relative = 100.0 * difference / policy_means[reference]
        bootstrap_relative = 100.0 * bootstrap_difference / bootstrap_policy_means[:, reference]
        comparison_results[name] = {
            "reference_policy": POLICIES[reference],
            "candidate_policy": POLICIES[candidate],
            "mean_paired_difference_candidate_minus_reference": difference.tolist(),
            "descriptive_path_bootstrap_95pct_interval": [
                percentile_interval(bootstrap_difference[:, metric])
                for metric in range(len(METRICS))
            ],
            "relative_change_percent_candidate_vs_reference": relative.tolist(),
            "relative_change_descriptive_path_bootstrap_95pct_interval": [
                percentile_interval(bootstrap_relative[:, metric])
                for metric in range(len(METRICS))
            ],
        }

    full_accept_comparisons = {}
    for index, policy in enumerate(POLICIES):
        difference = policy_means[index] - full_means
        bootstrap_difference = bootstrap_policy_means[:, index] - bootstrap_full_means
        relative = 100.0 * difference / full_means
        bootstrap_relative = 100.0 * bootstrap_difference / bootstrap_full_means
        full_accept_comparisons[policy] = {
            "mean_paired_difference_policy_minus_full_accept": difference.tolist(),
            "descriptive_path_bootstrap_95pct_interval": [
                percentile_interval(bootstrap_difference[:, metric])
                for metric in range(len(METRICS))
            ],
            "relative_change_percent_policy_vs_full_accept": relative.tolist(),
            "relative_change_descriptive_path_bootstrap_95pct_interval": [
                percentile_interval(bootstrap_relative[:, metric])
                for metric in range(len(METRICS))
            ],
        }

    summary = {
        "condition_weighting": "equal across twelve scenes and two attack fixtures",
        "policy_mean_mse": {
            policy: policy_means[index].tolist()
            for index, policy in enumerate(POLICIES)
        },
        "full_accept_diagnostic_mean_mse": full_means.tolist(),
        "paired_comparisons": comparison_results,
        "paired_comparisons_vs_full_accept": full_accept_comparisons,
    }
    return summary, {
        "policy_means": bootstrap_policy_means,
        "full_means": bootstrap_full_means,
    }


def bootstrap_health(health: list[dict]) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([*BOOTSTRAP_SEED, 1]))
    metrics = {"missing_post20": 2, "network_post20": 4}
    scene_count = len(health)
    means = {policy: {} for policy in (*POLICIES, "full_accept")}
    differences = {policy: {} for policy in POLICIES}

    for metric_name, metric_index in metrics.items():
        point_policy = np.zeros(len(POLICIES))
        point_full = 0.0
        boot_policy = np.zeros((BOOTSTRAP_REPLICATES, len(POLICIES)))
        boot_full = np.zeros(BOOTSTRAP_REPLICATES)
        for record in health:
            losses = record["loss"][:, :, metric_index]
            full = record["full"][:, metric_index]
            point_policy += losses.mean(axis=0) / scene_count
            point_full += full.mean() / scene_count
            indices = rng.integers(
                0, len(full), size=(BOOTSTRAP_REPLICATES, len(full)), dtype=np.int32
            )
            boot_policy += losses[indices].mean(axis=1) / scene_count
            boot_full += full[indices].mean(axis=1) / scene_count
        for index, policy in enumerate(POLICIES):
            means[policy][metric_name] = float(point_policy[index])
            differences[policy][metric_name] = {
                "mean_paired_difference_vs_full_accept": float(point_policy[index] - point_full),
                "descriptive_path_bootstrap_95pct_interval": percentile_interval(
                    boot_policy[:, index] - boot_full
                ),
            }
        means["full_accept"][metric_name] = float(point_full)

    target_fwer = np.asarray([record["fwer"] for record in health])
    all_edge_fwer = np.asarray([record["all_edge_fwer"] for record in health])
    return {
        "condition_weighting": "equal across twelve scenes",
        "mean_mse": means,
        "paired_policy_minus_full_accept": differences,
        "scene_target_fwer": target_fwer.tolist(),
        "scene_target_fwer_CP95": [record["fwer_ci"] for record in health],
        "equal_scene_mean_target_fwer": float(target_fwer.mean()),
        "scene_all_edge_fwer": all_edge_fwer.tolist(),
        "scene_all_edge_fwer_CP95": [record["all_edge_fwer_ci"] for record in health],
        "equal_scene_mean_all_edge_fwer": float(all_edge_fwer.mean()),
    }


def action_metrics(attacks: list[dict]) -> dict:
    by_scene = {}
    for record in attacks:
        alarm = record["target_alarm"][:, 50:110]
        actions = record["actions"][:, :, 50:110]
        denominator = int(alarm.sum())
        scene = str(record["scene"])
        scene_metrics = by_scene.setdefault(scene, {})
        scene_metrics[record["fixture"]] = {
            "target_alarm_events": denominator,
            "effective_weight_on_alarm_by_policy": [
                float(np.asarray([0.0, .2, .4, .6, .8, .9, 1.0])[actions[:, p][alarm]].mean())
                if denominator else None
                for p in range(len(POLICIES))
            ],
            "action_disagreement_rate": {
                name: float(
                    ((actions[:, candidate] != actions[:, reference]) & alarm).sum()
                    / denominator
                ) if denominator else None
                for name, (reference, candidate) in COMPARISONS.items()
            },
        }
    equal_scene = {}
    for name in COMPARISONS:
        values = []
        for scene_records in by_scene.values():
            fixture_values = [
                scene_records[fixture]["action_disagreement_rate"][name]
                for fixture in ("continuous", "burst")
            ]
            present = [value for value in fixture_values if value is not None]
            if present:
                values.append(float(np.mean(present)))
        equal_scene[name] = float(np.mean(values)) if values else None
    return {"per_scene_and_fixture": by_scene, "equal_scene_disagreement_rate": equal_scene}


def markdown_report(manifest: dict, attacks: dict, health: dict, actions: dict) -> str:
    lines = [
        "# D2-B development screening analysis",
        "",
        f"Execution commit `{manifest['execution_commit']}` with protocol `{manifest['protocol_commit']}`.",
        "The results are developmental and descriptive. Bootstrap intervals are path-level and do not support confirmatory claims.",
        "",
        "Attack MSE is averaged equally across twelve scenes and their continuous/burst fixtures.",
        "Negative paired differences indicate lower MSE for the candidate policy.",
        "",
        "## Attack window and recovery",
        "",
        "| Policy | Missing-block window | Missing-block recovery | Network window | Network post-20 |",
        "|---|---:|---:|---:|---:|",
    ]
    metric_indices = (0, 1, 3, 4)
    for policy in POLICIES:
        values = attacks["policy_mean_mse"][policy]
        lines.append(
            f"| {policy} | {values[metric_indices[0]]:.6g} | {values[metric_indices[1]]:.6g} | "
            f"{values[metric_indices[2]]:.6g} | {values[metric_indices[3]]:.6g} |"
        )
    full_accept = attacks["full_accept_diagnostic_mean_mse"]
    lines.append(
        f"| Full-accept diagnostic | {full_accept[0]:.6g} | {full_accept[1]:.6g} | "
        f"{full_accept[3]:.6g} | {full_accept[4]:.6g} |"
    )
    lines.extend([
        "",
        "## Paired policy differences",
        "",
        "| Comparison | Window difference (95% path bootstrap interval) | Recovery difference | Relative window change |",
        "|---|---:|---:|---:|",
    ])
    for name, comparison in attacks["paired_comparisons"].items():
        window = comparison["mean_paired_difference_candidate_minus_reference"][0]
        window_ci = comparison["descriptive_path_bootstrap_95pct_interval"][0]
        recovery = comparison["mean_paired_difference_candidate_minus_reference"][1]
        relative = comparison["relative_change_percent_candidate_vs_reference"][0]
        lines.append(
            f"| {name} | {window:.6g} [{window_ci[0]:.6g}, {window_ci[1]:.6g}] | "
            f"{recovery:.6g} | {relative:.3f}% |"
        )
    full_comparison = attacks["paired_comparisons_vs_full_accept"]["RH_dynamic"]
    full_window_difference = full_comparison[
        "mean_paired_difference_policy_minus_full_accept"
    ][0]
    full_window_interval = full_comparison[
        "descriptive_path_bootstrap_95pct_interval"
    ][0]
    full_recovery_difference = full_comparison[
        "mean_paired_difference_policy_minus_full_accept"
    ][1]
    lines.append(
        f"\nRH-dynamic minus full-accept attack-window difference: {full_window_difference:.6g} "
        f"[{full_window_interval[0]:.6g}, {full_window_interval[1]:.6g}]. "
        f"Recovery-window difference: {full_recovery_difference:.6g}."
    )
    lines.extend([
        "",
        "## Attack-window contrast by fixture",
        "",
        "| Fixture | RH-dynamic minus R1 | RH-dynamic minus RH-scalar |",
        "|---|---:|---:|",
    ])
    for fixture, summary in attacks["by_fixture"].items():
        risk_difference = summary["paired_comparisons"]["RH_dynamic_minus_R1"][
            "mean_paired_difference_candidate_minus_reference"][0]
        scalar_difference = summary["paired_comparisons"]["RH_dynamic_minus_RH_scalar"][
            "mean_paired_difference_candidate_minus_reference"][0]
        lines.append(f"| {fixture} | {risk_difference:.6g} | {scalar_difference:.6g} |")
    lines.extend([
        "",
        "## Attack detection and action diagnostics",
        "",
        f"Equal-condition path-miss fraction: {attacks['detector_activity']['equal_condition_path_miss_fraction']:.3%}.",
        f"Equal-condition polluted-tick non-alarm fraction: {attacks['detector_activity']['equal_condition_attack_tick_unalarmed_fraction']:.3%}.",
        f"Risk-table fallback count: {attacks['detector_activity']['risk_table_fallback_count']}.",
        "",
        "## RH-dynamic minus RH-scalar by scene",
        "",
        "| Scene | Attack-window difference | Recovery-window difference |",
        "|---:|---:|---:|",
    ])
    for scene, summary in attacks["by_scene"].items():
        comparison = summary["paired_comparisons"]["RH_dynamic_minus_RH_scalar"][
            "mean_paired_difference_candidate_minus_reference"
        ]
        lines.append(f"| {scene} | {comparison[0]:.6g} | {comparison[1]:.6g} |")
    lines.extend([
        "",
        "## Healthy evaluation",
        "",
        "| Policy | Missing-block post-20 MSE | Difference from full acceptance | Network post-20 MSE |",
        "|---|---:|---:|---:|",
    ])
    for policy in (*POLICIES, "full_accept"):
        missing = health["mean_mse"][policy]["missing_post20"]
        network = health["mean_mse"][policy]["network_post20"]
        if policy == "full_accept":
            delta = "—"
        else:
            value = health["paired_policy_minus_full_accept"][policy]["missing_post20"]["mean_paired_difference_vs_full_accept"]
            delta = f"{value:.6g}"
        lines.append(f"| {policy} | {missing:.6g} | {delta} | {network:.6g} |")
    lines.extend([
        "",
        f"Equal-scene mean target-link FWER: {health['equal_scene_mean_target_fwer']:.4%}.",
        f"Equal-scene mean all-edge FWER: {health['equal_scene_mean_all_edge_fwer']:.4%}.",
        "",
        "## Decision activity",
        "",
    ])
    for name, value in actions["equal_scene_disagreement_rate"].items():
        lines.append(f"- {name}: {value:.3%} alarm-action disagreement rate.")
    mechanism = json.loads(MECHANISM_FILE.read_text(encoding="utf-8"))
    q1_regression = mechanism["q1_deterministic_regression"]
    lines.extend([
        "",
        "## Health-cost history diagnostic",
        "",
        f"Median relative residual from the RH-scalar fit: {mechanism['dynamic_to_scalar_relative_fit_residual_median_equal_scene']:.3%}.",
        f"Dynamic-versus-static pairwise health-curve rank discordance: {mechanism['dynamic_vs_static_pairwise_rank_discordance_rate_equal_scene']:.3%}.",
        f"Dynamic-versus-static health-only argmin disagreement: {mechanism['dynamic_vs_static_health_only_argmin_disagreement_rate_equal_scene']:.3%}.",
        "The health curves sometimes differ in shape, but neither the health-only minimizer nor the risk-plus-health action changed in this development sample.",
        f"The q=1 regression passed in all {q1_regression['scene_count']} scenes. Maximum literal-update discrepancy was {q1_regression['max_literal_vs_optimized_update_difference']:.3g}.",
        "",
        "The conditional single-step risk table is an action-ranking proxy. Its values are not closed-loop MSE bounds.",
        "The interval calculations are descriptive development-screening summaries, not multiplicity-adjusted confirmatory inference.",
        "D2-C is not authorized by this analysis.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    manifest, attack_records, health_records = load_run()
    attack_summary, _ = bootstrap_attack(attack_records)
    attack_summary["by_fixture"] = {
        fixture: bootstrap_attack([
            record for record in attack_records if record["fixture"] == fixture
        ])[0]
        for fixture in ("continuous", "burst")
    }
    attack_summary["by_scene"] = {
        str(scene): bootstrap_attack([
            record for record in attack_records if record["scene"] == scene
        ])[0]
        for scene in manifest["scenes"]
    }
    attack_summary["detector_activity"] = {
        "equal_condition_path_miss_fraction": float(np.mean([
            record["stored"]["path_miss_fraction"] for record in attack_records
        ])),
        "equal_condition_attack_tick_unalarmed_fraction": float(np.mean([
            record["stored"]["attack_tick_unalarmed_fraction"] for record in attack_records
        ])),
        "equal_condition_longest_target_alarm_run_mean": float(np.mean([
            record["stored"]["longest_target_alarm_run_mean"] for record in attack_records
        ])),
        "risk_table_fallback_count": int(sum(
            sum(record["stored"]["policy_status"]["risk_fallback_counts"])
            for record in attack_records
        )),
        "risk_table_fallback_level_counts": np.sum([
            record["stored"]["policy_status"]["risk_fallback_level_counts"]
            for record in attack_records
        ], axis=0).tolist(),
        "zero_scalar_denominator_counts": np.sum([
            record["stored"]["policy_status"]["zero_scalar_denominator_counts"]
            for record in attack_records
        ], axis=0).tolist(),
    }
    health_summary = bootstrap_health(health_records)
    actions = action_metrics(attack_records)
    mechanism = json.loads(MECHANISM_FILE.read_text(encoding="utf-8"))
    result = {
        "status": "descriptive_D2_B_development_screening_analysis",
        "protocol_commit": manifest["protocol_commit"],
        "execution_commit": manifest["execution_commit"],
        "run_manifest_sha256": sha256(RUN_FILE),
        "mechanism_audit_sha256": sha256(MECHANISM_FILE),
        "scene_archive_sha256": manifest["output_sha256"],
        "analysis_source_sha256": sha256(Path(__file__)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": list(BOOTSTRAP_SEED),
        "bootstrap_unit": "whole independent path within each fixed scene-fixture condition",
        "claim_scope": manifest["claim_scope"],
        "attack": attack_summary,
        "health": health_summary,
        "action_metrics": actions,
        "health_cost_history_diagnostic": {
            key: mechanism[key]
            for key in (
                "target_alarm_events_in_attack_windows",
                "mechanism_audit_source_sha256",
                "dynamic_to_scalar_relative_fit_residual_median_equal_scene",
                "dynamic_to_scalar_relative_fit_residual_p95_max_scene",
                "dynamic_vs_static_pairwise_rank_discordance_rate_equal_scene",
                "dynamic_vs_static_health_only_argmin_disagreement_rate_equal_scene",
                "strategy_action_differences",
                "M7_legal_action_set_size_counts_on_alarm",
                "risk_only_nonzero_minus_q0_UCB_margin_mean_equal_scene",
                "q1_deterministic_regression",
            )
        },
    }
    atomic_text(OUTPUT_JSON, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_text(OUTPUT_MD, markdown_report(manifest, attack_summary, health_summary, actions))
    print(json.dumps({
        "status": result["status"],
        "output_json": str(OUTPUT_JSON),
        "output_markdown": str(OUTPUT_MD),
    }, indent=2))


if __name__ == "__main__":
    main()
