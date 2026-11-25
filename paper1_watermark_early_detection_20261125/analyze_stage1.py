"""Scenario-stratified paired-path analysis for Stage-1 attribution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution

HERE = Path(__file__).resolve().parent
ARMS = ("A_plugin_iid", "B_plugin_negative", "C_matched_iid",
        "D_matched_negative", "E_iid_wrong_key", "F_negative_wrong_key")
PRIMARY = "early3_per_link"
CONTRASTS = {
    "B_minus_A_plugin_watermark_gain": ("B_plugin_negative", "A_plugin_iid"),
    "D_minus_C_matched_watermark_gain": ("D_matched_negative", "C_matched_iid"),
    "interaction_matched_minus_plugin": ("D_minus_C", "B_minus_A"),
    "C_minus_E_iid_key_match_value": ("C_matched_iid", "E_iid_wrong_key"),
    "D_minus_F_negative_key_match_value": ("D_matched_negative", "F_negative_wrong_key"),
}
NBOOT = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clopper_pearson(successes, trials):
    return [
        float(beta_distribution.ppf(.025, successes, trials - successes + 1)) if successes else 0.,
        float(beta_distribution.ppf(.975, successes + 1, trials - successes)) if successes < trials else 1.,
    ]


def bootstrap_paired_scenarios(path_values, reps, seed):
    """Equal scenario weights; resample whole paths within each scenario."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    scenario_count = len(path_values)
    path_count = len(path_values[0])
    bootstrap_means = np.zeros((reps, scenario_count), dtype=float)
    point_means = np.zeros(scenario_count, dtype=float)
    for scenario, values in enumerate(path_values):
        values = np.asarray(values, dtype=float)
        if values.shape != (path_count,):
            raise ValueError("All scenario path contrasts must share one registered path count.")
        point_means[scenario] = values.mean()
        indices = rng.integers(0, path_count, size=(reps, path_count), dtype=np.int32)
        bootstrap_means[:, scenario] = values[indices].mean(axis=1)
    bootstrap_overall = bootstrap_means.mean(axis=1)
    return float(point_means.mean()), bootstrap_overall


def load_split(split_name):
    directory = HERE / f"stage1_{split_name}"
    manifest_path = directory / "RUN.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing complete {split_name} Stage-1 run")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != f"complete_{split_name}":
        raise ValueError(f"Stage-1 {split_name} manifest is incomplete")
    for filename, expected in manifest["archive_sha256"].items():
        if sha(directory / filename) != expected:
            raise ValueError(f"Path archive hash mismatch: {filename}")
    expected_scenarios = {row["name"] for row in manifest["scenario_specs"]}
    observed_scenarios = {
        name for record in manifest["models"].values()
        for name in record["attack_scenarios"]
    }
    if observed_scenarios != expected_scenarios:
        raise ValueError("Recorded scenario set differs from the fixed 12-scenario registration.")
    return directory, manifest


def analyze_attack(split_name, bootstrap_seed):
    directory, manifest = load_split(split_name)
    path_values = {metric: {arm: [] for arm in ARMS}
                   for metric in ("early1_per_link", "early2_per_link", "early3_per_link",
                                  "early5_per_link", "early8_per_link")}
    attack_loss_values = {arm: {metric: [] for metric in (
        "post20_network_mse", "post20_affected_mse", "attack_network_mse",
        "postattack_network_mse")}
        for arm in ARMS}
    scenario_records = {}
    for scenario in manifest["scenario_specs"]:
        name = scenario["name"]
        path = directory / f"{name}.npz"
        record = manifest["models"][scenario["model"]]["attack_scenarios"][name]
        with np.load(path, allow_pickle=False) as archive:
            scenario_values = {}
            for arm in ARMS:
                for metric in path_values:
                    values = archive[f"{arm}_{metric}"]
                    if values.shape[1] != 6:
                        raise ValueError(f"Per-link endpoint array has incorrect shape {name}:{arm}:{metric}")
                    path_value = values.mean(axis=1)
                    path_values[metric][arm].append(path_value)
                    scenario_values.setdefault(arm, {})[metric] = float(path_value.mean())
                for metric in attack_loss_values[arm]:
                    values = archive[f"{arm}_{metric}"]
                    path_values[metric][arm].append(values)
                    scenario_values.setdefault(arm, {})[metric] = float(values.mean())
        manifest_arm_metrics = manifest["models"][scenario["model"]][
            "attack_scenarios"
        ][name]["arms"]
        for arm in ARMS:
            for metric in ("path_any_link_early3", "path_any_link_full_attack",
                           "path_pre_onset_any_alarm", "eligible_link_time_alarm_fraction"):
                scenario_values.setdefault(arm, {})[metric] = manifest_arm_metrics[arm]["detection"][metric]["mean"]
        scenario_records[name] = {
            "model": scenario["model"],
            "angle_degrees": scenario["angle"],
            "onset": scenario["onset"],
            "paths": record["path_count"],
            "arms": scenario_values,
        }

    contrast_results = {}
    bootstrap_by_metric = {}
    for metric, methods in path_values.items():
        for arm in ARMS:
            mean, samples = bootstrap_paired_scenarios(
                path_values[metric][arm], NBOOT,
                (*bootstrap_seed, 1, len(bootstrap_by_metric), len(arm)),
            )
            bootstrap_by_metric[(metric, arm)] = (mean, samples)
        a_values = path_values[metric]["A_plugin_iid"]
        b_values = path_values[metric]["B_plugin_negative"]
        c_values = path_values[metric]["C_matched_iid"]
        d_values = path_values[metric]["D_matched_negative"]
        e_values = path_values[metric]["E_iid_wrong_key"]
        f_values = path_values[metric]["F_negative_wrong_key"]
        contrasts = {
            "B_minus_A_plugin_watermark_gain": [b - a for a, b in zip(a_values, b_values)],
            "D_minus_C_matched_watermark_gain": [d - c for c, d in zip(c_values, d_values)],
            "interaction_matched_minus_plugin": [
                (d - c) - (b - a)
                for a, b, c, d in zip(a_values, b_values, c_values, d_values)
            ],
            "C_minus_E_iid_key_match_value": [c - e for c, e in zip(c_values, e_values)],
            "D_minus_F_negative_key_match_value": [d - f for d, f in zip(d_values, f_values)],
        }
        contrast_results[metric] = {}
        for index, (name, values) in enumerate(contrasts.items()):
            mean, samples = bootstrap_paired_scenarios(
                values, NBOOT, (*bootstrap_seed, 2, list(path_values).index(metric), index)
            )
            contrast_results[metric][name] = {
                "mean_difference": mean,
                "descriptive_scenario_stratified_path_bootstrap_95": [
                    float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
                ],
                "scenario_means": {
                    scenario["name"]: float(value.mean())
                    for scenario, value in zip(manifest["scenario_specs"], values)
                },
            }

    loss_contrasts = {}
    for metric, methods in attack_loss_values.items():
        comparisons = {}
        for label, left, right in (
            ("B_minus_A", "B_plugin_negative", "A_plugin_iid"),
            ("D_minus_C", "D_matched_negative", "C_matched_iid"),
            ("interaction", "D_matched_negative", "C_matched_iid"),
            ("E_minus_C_wrong_key_minus_matched_iid", "E_iid_wrong_key", "C_matched_iid"),
            ("F_minus_D_wrong_key_minus_matched_negative", "F_negative_wrong_key", "D_matched_negative"),
        ):
            left_values = attack_loss_values[left][metric]
            right_values = attack_loss_values[right][metric]
            differences = [a - b for a, b in zip(left_values, right_values)]
            if label == "interaction":
                ba = [a - b for a, b in zip(
                    attack_loss_values["B_plugin_negative"][metric],
                    attack_loss_values["A_plugin_iid"][metric],
                )]
                differences = [d - c - b for d, c, b in zip(
                    attack_loss_values[left][metric],
                    attack_loss_values[right][metric], ba
                )]
            mean, samples = bootstrap_paired_scenarios(
                differences, NBOOT,
                (*bootstrap_seed, 3, len(loss_contrasts), len(comparisons)),
            )
            comparisons[label] = {
                "mean_difference": mean,
                "descriptive_scenario_stratified_path_bootstrap_95": [
                    float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
                ],
            }
        loss_contrasts[metric] = comparisons

    arm_means = {}
    runtime_means = {}
    for arm in ARMS:
        arm_means[arm] = {
            "early3_per_link_detection": float(np.mean([
                np.mean(values) for values in path_values["early3_per_link"][arm]
            ])),
            "early3_any_link_path_probability": float(np.mean([
                scenario_records[scenario["name"]]["arms"][arm]["path_any_link_early3"]
                ["mean"]
                for scenario in manifest["scenario_specs"]
            ])),
            "full_attack_no_alarm_fraction_per_link": float(np.mean([
                scenario_records[scenario["name"]]["arms"][arm]["no_alarm_fraction_per_link_full_attack"]
                for scenario in manifest["scenario_specs"]
            ])),
            "post20_network_mse": float(np.mean([
                row["arms"][arm]["post20_network_mse"]
                for row in scenario_records.values()
            ])),
            "post20_affected_mse": float(np.mean([
                row["arms"][arm]["post20_affected_mse"]
                for row in scenario_records.values()
            ])),
            "attack_network_mse": float(np.mean([
                row["arms"][arm]["attack_network_mse"]
                for row in scenario_records.values()
            ])),
        }
        runtime_means[arm] = {
            "score_seconds_per_scenario": float(np.mean([
                manifest["models"][scenario["model"]]["attack_scenarios"][scenario["name"]]["arms"][arm]["score_runtime_seconds"]
                for scenario in manifest["scenario_specs"]
            ])),
            "common_g_replay_seconds_per_scenario_including_source_plugin_score": float(np.mean([
                manifest["models"][scenario["model"]]["attack_scenarios"][scenario["name"]]["arms"][arm]["common_g_replay_wall_seconds_including_source_plugin_score"]
                for scenario in manifest["scenario_specs"]
            ])),
        }
    return {
        "status": f"descriptive_{split_name}_paired_path_analysis",
        "split": split_name,
        "baseline_commit": manifest["baseline_commit"],
        "execution_commit": manifest["execution_commit"],
        "run_manifest_sha256": sha(directory / "RUN.json"),
        "bootstrap_replicates": NBOOT,
        "bootstrap_seed": list(bootstrap_seed),
        "bootstrap_unit": "whole path within each fixed model-angle-onset scenario; scenarios equally weighted",
        "primary_endpoint": "early3_per_link",
        "detection_contrasts": contrast_results,
        "attack_loss_contrasts": loss_contrasts,
        "arm_equal_scenario_means": arm_means,
        "arm_runtime_equal_scenario_means": runtime_means,
        "healthy_holdout": {
            model: manifest["models"][model]["healthy_holdout"]["arms"]
            for model in MODELS
        },
        "scenarios": scenario_records,
        "claim_scope": manifest["claim_scope"],
    }


def render_report(results):
    primary = results["detection_contrasts"]["early3_per_link"]
    lines = [
        f"# Stage-1 {results['split']} score-attribution report",
        "",
        f"Frozen baseline `{results['baseline_commit']}`; experiment commit `{results['execution_commit']}`.",
        "Path bootstrap intervals resample whole physical paths within each of 12 fixed scenarios and weight scenarios equally.",
        "",
        "## Primary endpoint",
        "",
        "| Contrast in early three-attacked-report per-link detection probability | Mean difference | Descriptive 95% interval |",
        "|---|---:|---:|",
    ]
    for name in (
        "B_minus_A_plugin_watermark_gain",
        "D_minus_C_matched_watermark_gain",
        "interaction_matched_minus_plugin",
        "C_minus_E_iid_key_match_value",
        "D_minus_F_negative_key_match_value",
    ):
        record = primary[name]
        ci = record["descriptive_scenario_stratified_path_bootstrap_95"]
        lines.append(f"| {name} | {record['mean_difference']:.4%} | [{ci[0]:.4%}, {ci[1]:.4%}] |")
    lines.extend([
        "",
        "## Arm summaries",
        "",
        "| Arm | Early3 per-link detection | Any-link early3 path probability | Post20 network MSE | Post20 affected MSE | Attack network MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        values = results["arm_equal_scenario_means"][arm]
        lines.append(
            f"| {arm} | {values['early3_per_link_detection']:.3%} | "
            f"{values['early3_any_link_path_probability']:.3%} | "
            f"{values['post20_network_mse']:.6g} | {values['post20_affected_mse']:.6g} | "
            f"{values['attack_network_mse']:.6g} |"
        )
    lines.extend([
        "",
        "Per-link no-alarm fraction over the full 60-report attack window is reported below. First-alarm delays retain the 60-report no-alarm censoring value in the machine-readable paths.",
        "",
        "| Arm | No-alarm link fraction |",
        "|---|---:|",
    ])
    for arm in ARMS:
        values = results["arm_equal_scenario_means"][arm]
        lines.append(f"| {arm} | {values['full_attack_no_alarm_fraction_per_link']:.3%} |")
    lines.extend([
        "",
        "## Other detection endpoints",
        "",
        "The JSON report includes first one/two/five/eight attacked-report endpoints, whole-path early/attack detection, first-alarm delay with 60-tick censoring, pre-onset alarms, and eligible link-time rates.",
        "",
        "## Healthy holdout",
        "",
        "| Model | Arm | FWER | Exact 95% interval | Post-20 network MSE |",
        "|---|---|---:|---:|---:|",
    ])
    for model in MODELS:
        for arm, record in results["healthy_holdout"][model].items():
            ci = record["Clopper_Pearson_95"]
            lines.append(f"| {model} | {arm} | {record['FWER']:.3%} | [{ci[0]:.3%}, {ci[1]:.3%}] | {record['post20_network_mse_mean']:.6g} |")
    lines.extend([
        "",
        "## Runtime",
        "",
        "| Arm | Score seconds/scenario | Common-G replay seconds/scenario, including its frozen plugin score pass |",
        "|---|---:|---:|",
    ])
    for arm in ARMS:
        values = results["arm_runtime_equal_scenario_means"][arm]
        lines.append(f"| {arm} | {values['score_seconds_per_scenario']:.6g} | {values['common_g_replay_seconds_per_scenario_including_source_plugin_score']:.6g} |")
    lines.extend([
        "",
        "## Estimation metrics",
        "",
        "The JSON report includes paired network and affected-receiver post-20, attack-window and postattack MSE contrasts for B−A, D−C, their interaction and matched-key versus same-law wrong-key controls.",
        "",
        "This experiment covers the frozen six-link support, two-dimensional messages and common-G estimator. It does not validate unknown-support detection or an information-theoretic secrecy guarantee.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    args = parser.parse_args()
    bootstrap_seed = (20261125, 8317, 100 if args.split == "development" else 200)
    result = analyze_attack(args.split, bootstrap_seed)
    directory = HERE / SPLITS[args.split]["directory"]
    result["analysis_source_sha256"] = digest(Path(__file__))
    json_path = directory / "ANALYSIS.json"
    md_path = directory / "ANALYSIS.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    md_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
