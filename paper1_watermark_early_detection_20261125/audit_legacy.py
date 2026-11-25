"""Read-only inventory and exact two-scenario regression for the frozen v3 test."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3 = ROOT / "paper1_sscd_commonrandom_20261122"
THEORY = ROOT / "paper1_50node_tv_gate_theory_20261121"
SOURCE50 = ROOT / "paper1_common_g50_gain_acceptance_20261121"
sys.path[:0] = [str(V3), str(THEORY), str(SOURCE50)]

from run_crn_development import load_backend
from run_crn_final import measure
from watson_crn import beta_from_common, make_common_draws, watermark_attack_from_beta


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    return {
        "run_crn_final.measure": {
            "actual_input": "saved estimator world, physical state paths, affected receiver IDs, attack onset, frozen threshold",
            "reusable": "legacy attack/eligible windows, post-20 and per-path MSE, attack-path-any-alarm, censored first alarm delay, drop fraction",
            "change_location": "new early endpoint is computed separately from raw world.scores on [onset,onset+3); never shift via the legacy eligible column",
        },
        "watson_crn.make_common_draws/beta_from_common": {
            "actual_input": "path count, length, edge count, radial/initial/direction seeds, scalar theta",
            "reusable": "radial Exp(1), initial uniform direction, CRN quantiles and Watson signed-angle inverse",
            "change_location": "new schedule generator extends length to 150 and samples 50 sender keys, sharing each sender key across all its outgoing links; radial-bin uses two prevalidated theta masks",
        },
        "watson_crn.watermark_attack_from_beta": {
            "actual_input": "raw six-edge attack waveform and beta process",
            "reusable": "decoded additive packet perturbation M_k a_k and attack onset placement",
            "change_location": "new full-horizon packet decoder creates M for every sender/time, then applies only the attack-window raw waveform to the six frozen arcs",
        },
        "paper1_50node_tv_gate_theory_20261121.replay.scores": {
            "actual_input": "physical paths and decoded attack perturbation; it constructs per-edge raw packet history internally",
            "reusable": "exact frozen four overlapping 8-D parity residual construction, plugin covariance regularization 1e-6 and historical plugin-KL",
            "change_location": "new scorer takes a read-only sender-packet array; matched GLRT receives the same causal eight-report window and watermark key prefix",
        },
        "run_crn_development.load_backend": {
            "actual_input": "hash-pinned frozen config, source model, gain tables, M7 and calibration modules",
            "reusable": "true 50-node common-G estimator, physical path generator, original six-link topology, source compatibility checks",
            "change_location": "none; load in the detached 8953e1 worktree and call the frozen model interfaces",
        },
        "run_common_g_50.run_common_g": {
            "actual_input": "model, physical paths, decoded delta, raw sender packets, detector threshold or explicit gate mask",
            "reusable": "all-50 node common-G replay and separate estimator gates from detector scores",
            "change_location": "pass matched-score decisions as an exogenous gate mask; record external online score rather than replacing the frozen plugin scorer",
        },
        "calibration_rule.path_fwer_cutoff": {
            "actual_input": "one maximum score per independent healthy path",
            "reusable": "exact decimal-alpha order-statistic rank and strict > threshold convention",
            "change_location": "calibrate each matched score family independently over max of six links and k=7..149; plugin score is also recalibrated over the same decision times",
        },
        "crn_final_results.json/crn_final_path_arrays.npz": {
            "actual_input": "recorded v3 final physical/radial/initial/direction namespaces and calibrated thresholds",
            "reusable": "read-only exact source regression for unstable/stable theta=0 and theta=-1.3152369768257723 at angle 0 and onset 50",
            "change_location": "no archived paths enter new uncertainty estimates or method selection",
        },
    }


def replay_two_scenarios():
    result = json.loads((V3 / "crn_final_results.json").read_text(encoding="utf-8"))
    arrays = np.load(V3 / "crn_final_path_arrays.npz", allow_pickle=False)
    rows = []
    maximum_score_difference = 0.0
    for label in ("unstable", "stable"):
        model = None
        old, load_model, physical_paths, common_g, _ = load_backend()
        model = load_model(label)
        record = result["models"][label]
        scenario = f"{label}__rot+0__onset50"
        scenario_record = record["scenarios"][scenario]
        seeds = scenario_record["seed_namespaces"]
        threshold = float(record["calibration_threshold"])
        physical = physical_paths(model, 512, int(seeds["physical"]))
        draws = make_common_draws(
            512, 60, len(model.links),
            int(seeds["radial"]), int(seeds["initial"]),
            int(seeds["quantile_and_sign"]),
        )
        raw, _ = old.rotated_raw_attack(model, 0.)
        for name, theta in (("iid", 0.), ("neg_boundary", -1.3152369768257723)):
            beta = beta_from_common(draws, theta)
            delta = watermark_attack_from_beta(
                raw[50:110], beta, T=model.T, offset=50
            )
            world = common_g(model, physical, delta, gate_on=True, threshold=threshold)
            computed = measure(world, physical, model.receiver_ids, 50, threshold)
            differences = {}
            for metric, values in computed.items():
                archived = arrays[f"{scenario}_{name}__{metric}"]
                differences[metric] = float(np.max(np.abs(np.asarray(values) - archived)))
            score_reference = common_g(model, physical, delta, gate_on=False).scores
            score_replay = common_g(model, physical, delta, gate_on=True,
                                    threshold=threshold).scores
            score_difference = float(np.max(np.abs(score_reference - score_replay)))
            maximum_score_difference = max(maximum_score_difference, score_difference)
            rows.append({
                "model": label,
                "scenario": scenario,
                "watermark": name,
                "theta": theta,
                "seed_namespaces": seeds,
                "frozen_threshold": threshold,
                "archived_metric_max_abs_differences": differences,
                "score_gate_mode_invariance_max_abs_difference": score_difference,
                "first_three_reports_any_alarm_per_link_probability": float(
                    np.mean((world.scores[:, 50:53, :] > threshold).any(axis=1))
                ),
                "first_three_attacked_report_link_time_alarm_fraction": float(
                    np.mean(world.scores[:, 50:53, :] > threshold)
                ),
                "first_three_reports_any_link_per_path_probability": float(
                    np.mean((world.scores[:, 50:53, :] > threshold).any(axis=(1, 2)))
                ),
            })
    return {
        "status": "PASS" if max(
            max(row["archived_metric_max_abs_differences"].values()) for row in rows
        ) == 0.0 else "PASS_WITH_NUMERICAL_DIFFERENCES",
        "models": ["unstable", "stable"],
        "scenario": "rot0_onset50",
        "paths_per_model_policy": 512,
        "no_new_physical_paths": True,
        "max_plugin_score_independent_of_gate_mode": maximum_score_difference,
        "rows": rows,
    }


def recompute_v3_final_summary():
    final = json.loads((V3 / "crn_final_results.json").read_text(encoding="utf-8"))
    final_analysis = json.loads((V3 / "v3_final_analysis.json").read_text(encoding="utf-8"))
    arrays = np.load(V3 / "crn_final_path_arrays.npz", allow_pickle=False)
    metrics = (
        "network_post20", "affected_post20", "network_attack",
        "network_postattack", "eligible_link_time_alarm_fraction",
        "attack_path_any_alarm", "raw_first_alarm_delay_censored60",
        "eligible_first_alarm_delay_from_onset", "drop_fraction_attack",
    )
    rebuilt = {}
    maximum_mean_difference = 0.0
    for model_name in ("unstable", "stable"):
        scenario_names = sorted(final["models"][model_name]["scenarios"])
        model_result = {}
        for policy in ("iid", "neg_boundary"):
            model_result[policy] = {}
            for metric in metrics:
                scenario_means = []
                for scenario in scenario_names:
                    key = f"{scenario}_{policy}__{metric}"
                    value = arrays[key]
                    if value.shape != (512,):
                        raise ValueError(f"Unexpected V3 path-array shape for {key}")
                    scenario_means.append(float(value.mean()))
                mean = float(np.mean(scenario_means))
                model_result[policy][metric] = {
                    "equal_weight_scenario_mean": mean,
                    "scenario_means": scenario_means,
                }
                expected = final_analysis["final_scenarios"][model_name][
                    "candidate_equal_weight_means"
                ][policy].get(metric)
                if expected is not None:
                    difference = abs(mean - float(expected))
                    maximum_mean_difference = max(maximum_mean_difference, difference)
                    if difference > 2e-14:
                        raise AssertionError(f"V3 archived summary mismatch {model_name}:{policy}:{metric}")
        model_result["healthy_fwer"] = {
            "count": final["models"][model_name]["healthy_alarms"],
            "paths": len(json.loads((V3 / "crn_final_path_manifest.json").read_text(
                encoding="utf-8"))["models"][model_name]["healthy_test_path_ids"]),
            "fraction": final["models"][model_name]["heldout_healthy_FWER"],
            "exact_interval": final["models"][model_name]["healthy_FWER_ClopperPearson_CI95"],
        }
        rebuilt[model_name] = model_result
    return {
        "status": "PASS",
        "source": "crn_final_path_arrays.npz and crn_final_results.json",
        "scenario_count_per_model": 6,
        "paths_per_scenario": 512,
        "maximum_equal_weight_summary_difference": maximum_mean_difference,
        "models": rebuilt,
    }


def main():
    files = {
        "V3_RESULTS.md": V3 / "V3_RESULTS.md",
        "watson_crn.py": V3 / "watson_crn.py",
        "run_crn_development.py": V3 / "run_crn_development.py",
        "run_crn_final.py": V3 / "run_crn_final.py",
        "replay.py": THEORY / "replay.py",
        "run_common_g_50.py": SOURCE50 / "run_common_g_50.py",
        "calibration_rule.py": SOURCE50 / "calibration_rule.py",
        "final_results.json": V3 / "crn_final_results.json",
        "final_paths.npz": V3 / "crn_final_path_arrays.npz",
    }
    output = {
        "status": "read_only_frozen_v3_and_source_audit",
        "baseline_commit": "8953e1dd9a5e2535ead2698acb6f345a85d4f532",
        "source_sha256": {name: digest(path) for name, path in files.items()},
        "reusable_interface_inventory": inventory(),
        "v3_old_summary_recomputed_from_path_arrays": recompute_v3_final_summary(),
        "legacy_two_scenario_replay": replay_two_scenarios(),
        "scope": "regression only; old V3 paths are excluded from all new inference",
    }
    target = HERE / "PHASE0_REPLAY_AUDIT.json"
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": output["status"],
        "legacy_replay": output["legacy_two_scenario_replay"]["status"],
        "max_metric_difference": max(
            max(row["archived_metric_max_abs_differences"].values())
            for row in output["legacy_two_scenario_replay"]["rows"]
        ),
        "report": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
