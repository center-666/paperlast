"""Paired descriptive analysis of the constrained-objective D2-B2 screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_b as common_analysis

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "b2_results"
RUN_FILE = RUN_DIR / "B2_RUN.json"
OUTPUT_JSON = RUN_DIR / "B2_ANALYSIS.json"
OUTPUT_MD = RUN_DIR / "B2_ANALYSIS.md"
POLICIES = ("R1", "HC_static", "HC_scalar", "HC_dynamic")
COMPARISONS = {
    "HC_dynamic_minus_R1": (0, 3),
    "HC_dynamic_minus_HC_static": (1, 3),
    "HC_dynamic_minus_HC_scalar": (2, 3),
}
common_analysis.POLICIES = POLICIES
common_analysis.COMPARISONS = COMPARISONS
common_analysis.BOOTSTRAP_SEED = (20261125, 7319, 2, 992)
common_analysis.BOOTSTRAP_REPLICATES = 2048


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run():
    manifest = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_D2_B2_development_screening":
        raise ValueError("Analysis requires all twelve completed D2-B2 scenes.")
    if manifest.get("claim_scope") != "independent_development_screening_proxy_constraints_not_health_guarantees":
        raise ValueError("Unexpected D2-B2 evidence scope.")
    attacks, healthy = [], []
    for scene in manifest["results"]:
        filename = f"scene_{int(scene):02}.npz"
        path = RUN_DIR / filename
        if sha256(path) != manifest["output_sha256"].get(filename):
            raise ValueError(f"D2-B2 archive hash mismatch: {filename}")
        result = manifest["results"][scene]
        with np.load(path, allow_pickle=False) as archive:
            healthy.append({
                "scene": int(scene),
                "loss": archive["health_loss"].copy(),
                "full": archive["health_full_accept_loss"].copy(),
                "fwer": result["target_fwer"],
                "fwer_ci": result["target_fwer_CP95"],
                "all_edge_fwer": result["all_edge_fwer"],
                "all_edge_fwer_ci": result["all_edge_fwer_CP95"],
            })
            for fixture in ("continuous", "burst"):
                loss = archive[f"{fixture}_loss"].copy()
                full = archive[f"{fixture}_full_accept_loss"].copy()
                actions = archive[f"{fixture}_actions"].copy()
                masks = archive[f"{fixture}_masks"].copy()
                feasible = archive[f"{fixture}_feasible_sets"].copy()
                utilization = archive[f"{fixture}_budget_utilization"].copy()
                np.testing.assert_allclose(
                    loss.mean(axis=0),
                    np.asarray(result["attacks"][fixture]["loss_mean"]),
                    rtol=1e-12, atol=1e-12,
                )
                attacks.append({
                    "scene": int(scene),
                    "fixture": fixture,
                    "loss": loss,
                    "full": full,
                    "actions": actions,
                    "feasible": feasible,
                    "budget_utilization": utilization,
                    "target_alarm": masks[:, :, 6].astype(bool),
                    "stored": result["attacks"][fixture],
                })
    return manifest, attacks, healthy


def summarize_attacks(records):
    full_summary, boot = common_analysis.bootstrap_attack(records)
    by_scene = {}
    for scene in sorted({record["scene"] for record in records}):
        by_scene[str(scene)] = common_analysis.bootstrap_attack(
            [record for record in records if record["scene"] == scene]
        )[0]
    by_fixture = {
        fixture: common_analysis.bootstrap_attack(
            [record for record in records if record["fixture"] == fixture]
        )[0]
        for fixture in ("continuous", "burst")
    }
    full_summary["by_scene"] = by_scene
    full_summary["by_fixture"] = by_fixture
    full_summary["detector_activity"] = {
        "equal_condition_path_miss_fraction": float(np.mean([
            record["stored"]["path_miss_fraction"] for record in records
        ])),
        "equal_condition_attack_tick_unalarmed_fraction": float(np.mean([
            record["stored"]["attack_tick_unalarmed_fraction"] for record in records
        ])),
        "risk_table_fallback_count": int(sum(
            sum(record["stored"]["health_status"]["risk_fallback_counts"])
            for record in records
        )),
        "risk_table_fallback_level_counts": np.sum([
            record["stored"]["health_status"]["risk_fallback_level_counts"]
            for record in records
        ], axis=0).tolist(),
        "health_forced_out_candidate_counts": np.sum([
            record["stored"]["health_status"]["health_forced_out_candidate_counts"]
            for record in records
        ], axis=0).tolist(),
        "health_budget_active_decision_counts": np.sum([
            record["stored"]["health_status"]["health_budget_active_decision_counts"]
            for record in records
        ], axis=0).tolist(),
        "scalar_zero_denominator_counts": np.sum([
            record["stored"]["health_status"]["scalar_zero_denominator_counts"]
            for record in records
        ], axis=0).tolist(),
    }
    return full_summary


def health_constraint_activity(records):
    per_scene = {}
    feasible_count_total = np.zeros(4)
    alarm_total = 0
    q0_feasible_total = np.zeros(4)
    utilization_values = [[] for _ in range(4)]
    for record in records:
        mask = record["target_alarm"][:, 50:110]
        feasible = record["feasible"][:, :, 50:110]
        utilization = record["budget_utilization"][:, :, 50:110]
        scene = str(record["scene"])
        fixture = record["fixture"]
        alarm_count = int(mask.sum())
        feasible_sizes = feasible.sum(axis=-1)
        q0_legal = feasible[..., 0]
        size_sums = np.sum(feasible_sizes * mask[:, None, :], axis=(0, 2))
        q0_counts = np.sum(q0_legal * mask[:, None, :], axis=(0, 2))
        feasible_count_total += size_sums
        q0_feasible_total += q0_counts
        alarm_total += alarm_count
        record_utilizations = []
        for policy in range(1, 4):
            values = utilization[:, policy][mask]
            utilization_values[policy].extend(values.tolist())
            record_utilizations.append(float(np.mean(values)) if len(values) else None)
        per_scene.setdefault(scene, {})[fixture] = {
            "target_alarm_events": alarm_count,
            "mean_health_feasible_action_count_by_policy": [
                float(size_sums[index] / alarm_count) if alarm_count else None
                for index in range(4)
            ],
            "q0_health_feasible_fraction_by_policy": [
                float(q0_counts[index] / alarm_count) if alarm_count else None
                for index in range(4)
            ],
            "mean_budget_utilization_HC_static_scalar_dynamic": record_utilizations,
        }
    return {
        "scene_fixture": per_scene,
        "target_alarm_events": alarm_total,
        "mean_health_feasible_action_count_by_policy": (
            feasible_count_total / alarm_total
        ).tolist() if alarm_total else [None] * 4,
        "q0_health_feasible_fraction_by_policy": (
            q0_feasible_total / alarm_total
        ).tolist() if alarm_total else [None] * 4,
        "mean_budget_utilization_HC_static_scalar_dynamic": [
            float(np.mean(utilization_values[index])) if utilization_values[index] else None
            for index in range(1, 4)
        ],
    }


def render_markdown(manifest, attack, healthy, action):
    constraint = attack["health_constraint_activity"]
    names = POLICIES
    metrics = (0, 1, 3, 4)
    lines = [
        "# D2-B2 constrained-objective development screen",
        "",
        f"Execution commit `{manifest['execution_commit']}`.",
        "This is independent development-screening evidence. Nominal-health feasibility is not a formal adaptive-alarm health guarantee.",
        "",
        "Attack MSE averages equally over 12 scenes and both attack fixtures. Negative paired differences favor the candidate policy.",
        "",
        "## Attack and recovery MSE",
        "",
        "| Policy | Missing block, attack | Missing block, recovery | Network, attack | Network, post-20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in names:
        value = attack["policy_mean_mse"][policy]
        lines.append("| {} | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
            policy, *(value[index] for index in metrics)
        ))
    full = attack["full_accept_diagnostic_mean_mse"]
    lines.append("| Full-accept diagnostic | {:.6g} | {:.6g} | {:.6g} | {:.6g} |".format(
        *(full[index] for index in metrics)
    ))
    lines.extend([
        "",
        "## Paired attack-window contrasts",
        "",
        "| Comparison | Difference, 95% path-bootstrap interval | Relative change |",
        "|---|---:|---:|",
    ])
    for name, record in attack["paired_comparisons"].items():
        difference = record["mean_paired_difference_candidate_minus_reference"][0]
        interval = record["descriptive_path_bootstrap_95pct_interval"][0]
        relative = record["relative_change_percent_candidate_vs_reference"][0]
        lines.append(f"| {name} | {difference:.6g} [{interval[0]:.6g}, {interval[1]:.6g}] | {relative:.3f}% |")
    lines.extend([
        "",
        "## Per-scene HC-dynamic minus HC-scalar",
        "",
        "| Scene | Attack window | Recovery window |",
        "|---:|---:|---:|",
    ])
    for scene, row in attack["by_scene"].items():
        values = row["paired_comparisons"]["HC_dynamic_minus_HC_scalar"][
            "mean_paired_difference_candidate_minus_reference"
        ]
        lines.append(f"| {scene} | {values[0]:.6g} | {values[1]:.6g} |")
    lines.extend([
        "",
        "## Healthy evaluation",
        "",
        "| Policy | Missing-block post-20 MSE | Paired difference vs full acceptance |",
        "|---|---:|---:|",
    ])
    for policy in (*names, "full_accept"):
        value = healthy["mean_mse"][policy]["missing_post20"]
        difference = ("—" if policy == "full_accept" else
                      f"{healthy['paired_policy_minus_full_accept'][policy]['missing_post20']['mean_paired_difference_vs_full_accept']:.6g}")
        lines.append(f"| {policy} | {value:.6g} | {difference} |")
    lines.extend([
        "",
        f"Equal-scene target-link FWER: {healthy['equal_scene_mean_target_fwer']:.3%}.",
        f"Equal-scene all-edge FWER: {healthy['equal_scene_mean_all_edge_fwer']:.3%}.",
        f"Equal-condition path-miss fraction: {attack['detector_activity']['equal_condition_path_miss_fraction']:.3%}.",
        f"Equal-condition polluted-tick non-alarm fraction: {attack['detector_activity']['equal_condition_attack_tick_unalarmed_fraction']:.3%}.",
        f"Risk-table fallback count: {attack['detector_activity']['risk_table_fallback_count']}.",
        f"Mean health-feasible action counts R1/static/scalar/dynamic: {constraint['mean_health_feasible_action_count_by_policy']}.",
        f"q=0 feasible fractions R1/static/scalar/dynamic: {constraint['q0_health_feasible_fraction_by_policy']}.",
        f"Mean nominal-health budget utilization static/scalar/dynamic: {constraint['mean_budget_utilization_HC_static_scalar_dynamic']}.",
        "",
        "Health-budget active decisions for HC-static, HC-scalar and HC-dynamic: "
        + str(attack["detector_activity"]["health_budget_active_decision_counts"]) + ".",
        "## Action changes",
        "",
    ])
    for name, value in action["equal_scene_disagreement_rate"].items():
        lines.append(f"- {name}: {value:.3%} alarm-decision disagreement.")
    lines.extend([
        "",
        "The risk table is a conditional incremental-damage proxy, not a closed-loop MSE bound. All intervals are descriptive development-screening summaries.",
        "D2-C remains unstarted.",
        "",
    ])
    return "\n".join(lines)


def main():
    manifest, attack_records, healthy_records = load_run()
    attack = summarize_attacks(attack_records)
    attack["health_constraint_activity"] = health_constraint_activity(attack_records)
    healthy = common_analysis.bootstrap_health(healthy_records)
    action = common_analysis.action_metrics(attack_records)
    output = {
        "status": "descriptive_D2_B2_independent_development_screening_analysis",
        "execution_commit": manifest["execution_commit"],
        "protocol_commit": manifest["protocol_commit"],
        "B2_RUN_sha256": sha256(RUN_FILE),
        "scene_archive_sha256": manifest["output_sha256"],
        "analysis_source_sha256": sha256(Path(__file__)),
        "bootstrap_replicates": common_analysis.BOOTSTRAP_REPLICATES,
        "bootstrap_seed": list(common_analysis.BOOTSTRAP_SEED),
        "bootstrap_unit": "whole path within each frozen scene-fixture condition",
        "claim_scope": manifest["claim_scope"],
        "attack": attack,
        "health": healthy,
        "action_metrics": action,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    OUTPUT_MD.write_text(render_markdown(manifest, attack, healthy, action),
                         encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "analysis": str(OUTPUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
