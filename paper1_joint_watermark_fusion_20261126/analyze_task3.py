"""Paired path readout for task-3 development screening."""
import csv
import json
from pathlib import Path

import numpy as np

import models_quality as quality
import task3

HERE = Path(__file__).resolve().parent
ROOT = HERE / "task3_dev"
REPS = 20000
SEED = [20261126, 94017, 83, 2203]
METRICS = ("attack_full", "attack_active", "attack_inactive", "network_attack",
           "recovery", "health_post20", "network_post20",
           "nondefault_action_fraction", "q0_action_fraction",
           "maximum_missing_block_error_sq")
PAIRS = (
    ("T_fixed_table", "A_q0_raw"),
    ("T_fixed_table", "A_M7_min"),
    ("H_fixed_epsilon", "T_fixed_table"),
    ("V_updated_epsilon", "H_fixed_epsilon"),
    ("D0_duration", "V_updated_epsilon"),
    ("D1_match_score", "D0_duration"),
    ("D1_match_score", "V_updated_epsilon"),
    ("q1_full", "A_q0_raw"),
)


def csvout(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_path = ROOT / "RUN.json"
    run = quality.json.loads(run_path.read_text(encoding="utf-8"))
    if run["status"] != "complete_task3_development":
        raise ValueError("Task3 development is incomplete.")
    for filename, expected in run["archives"].items():
        assert task3.digest(ROOT / filename) == expected
    rng = np.random.default_rng(SEED)
    absolute, paired, scenes, health = [], [], [], []
    summaries = {}
    for config_index, name in enumerate(task3.QUALITY_ORDER):
        cfg = run["configs"][name]
        archive_path = ROOT / f"config{config_index:02}.npz"
        with np.load(archive_path) as archive:
            fixture_losses = {}
            for fixture in ("continuous", "burst"):
                losses = archive[f"{fixture}_losses"]
                gates = archive[f"{fixture}_gates"].astype(bool)
                actions = archive[f"{fixture}_actions"]
                fixture_losses[fixture] = losses
                for p, policy in enumerate(task3.POLICIES):
                    for m, metric in enumerate(METRICS):
                        values = losses[:, p, m]
                        scenes.append({
                            "configuration": name, "fixture": fixture,
                            "policy": policy, "metric": metric,
                            "mean": float(values.mean()),
                        })
                        absolute.append({
                            "configuration": name, "fixture": fixture,
                            "policy": policy, "metric": metric,
                            "mean": float(values.mean()), "median": float(np.median(values)),
                            "p99": float(np.quantile(values, .99)),
                            "max": float(values.max()),
                            "count_gt1": int(np.count_nonzero(values > 1)),
                        })
            for left, right in PAIRS:
                li, ri = task3.POLICIES.index(left), task3.POLICIES.index(right)
                for metric_index, metric in enumerate(METRICS):
                    differences = [fixture_losses[f][:, li, metric_index]
                                   - fixture_losses[f][:, ri, metric_index]
                                   for f in ("continuous", "burst")]
                    point = np.mean([x.mean() for x in differences])
                    draws = []
                    for values in differences:
                        n = len(values)
                        indices = rng.integers(0, n, size=(REPS, n))
                        draws.append(values[indices].mean(axis=1))
                    boot = np.mean(draws, axis=0)
                    paired.append({
                        "configuration": name, "comparison": left+"-"+right,
                        "metric": metric, "difference": float(point),
                        "lower95": float(np.quantile(boot, .025)),
                        "upper95": float(np.quantile(boot, .975)),
                        "bootstrap_replicates": REPS,
                    })
            for left, right in (
                ("T_fixed_table", "A_q0_raw"),
                ("T_fixed_table", "A_M7_min"),
                ("H_fixed_epsilon", "T_fixed_table"),
                ("V_updated_epsilon", "H_fixed_epsilon"),
                ("D0_duration", "V_updated_epsilon"),
                ("D1_match_score", "D0_duration"),
                ("D1_match_score", "V_updated_epsilon"),
            ):
                li, ri = task3.POLICIES.index(left), task3.POLICIES.index(right)
                path_differences = [
                    (archive[f"{fixture}_actions"][:, li, 50:110]
                     != archive[f"{fixture}_actions"][:, ri, 50:110]).mean(axis=1)
                    for fixture in ("continuous", "burst")
                ]
                samples = []
                for values in path_differences:
                    indices = rng.integers(0, len(values), size=(REPS, len(values)))
                    samples.append(values[indices].mean(axis=1))
                boot = (samples[0]+samples[1])/2
                paired.append({
                    "configuration": name, "comparison": left+"-"+right,
                    "metric": "path_action_disagreement_fraction",
                    "difference": float((path_differences[0].mean()
                                         + path_differences[1].mean())/2),
                    "lower95": float(np.quantile(boot, .025)),
                    "upper95": float(np.quantile(boot, .975)),
                    "bootstrap_replicates": REPS,
                })
            health_loss = archive["health_losses"]
            health_gates = archive["health_gates"].astype(bool)
            health_actions = archive["health_actions"]
            for policy_index, policy in enumerate(task3.POLICIES):
                clear_counts = []
                first_clear_weights = []
                for path in range(len(health_gates)):
                    target_alarm = health_gates[path, :, 6]
                    clear = np.flatnonzero(target_alarm[:-1] & ~target_alarm[1:])+1
                    clear_counts.append(len(clear))
                    if len(clear):
                        first_clear_weights.extend(
                            quality.Q[np.asarray(health_actions[path, policy_index])[clear]])
                health.append({
                    "configuration": name, "policy": policy,
                    "metric": "health_false_alarm_clear_and_reaccept",
                    "paths": len(health_gates), "clear_events": int(sum(clear_counts)),
                    "mean": None, "upper_quantile99": None,
                    "mean_first_clear_weight": float(np.mean(first_clear_weights))
                    if first_clear_weights else None,
                    "first_clear_weight_is_q1_by_registered_immediate_recovery":
                    bool(not first_clear_weights or np.all(np.asarray(first_clear_weights) == 1.)),
                })
            for policy_index, policy in enumerate(task3.POLICIES):
                for metric_index, metric in ((5, "health_missing_post20"),
                                             (6, "health_network_post20")):
                    values = health_loss[:, policy_index, metric_index]
                    health.append({
                        "configuration": name, "policy": policy, "metric": metric,
                        "mean": float(values.mean()), "paths": len(values),
                        "clear_events": None, "mean_first_clear_weight": None,
                        "upper_quantile99": float(np.quantile(values, .99)),
                    })
            summaries[name] = cfg["risk_and_loss"]
    for file, rows in (("absolute.csv", absolute), ("paired.csv", paired),
                       ("scenes.csv", scenes), ("health.csv", health)):
        csvout(ROOT / file, rows)
    report = {
        "status": "complete_task3_development_analysis",
        "run_sha256": task3.digest(run_path), "source_sha256": task3.digest(Path(__file__)),
        "bootstrap_replicates": REPS, "bootstrap_seed": SEED,
        "unit": "Independent physical path within fixed configuration and fixture, paired across policies.",
        "fixtures_equal_weighted": True,
        "interval_scope": "Pointwise path-paired descriptive bootstrap. Development only.",
        "tables_sha256": {f: task3.digest(ROOT/f) for f in
                          ("absolute.csv", "paired.csv", "scenes.csv", "health.csv")},
        "configuration_summaries": summaries,
    }
    quality.frozen.atomic_json(ROOT / "ANALYSIS.json", report)
    print(json.dumps({"status": report["status"], "absolute_rows": len(absolute),
                      "paired_rows": len(paired), "health_rows": len(health)}))


if __name__ == "__main__":
    main()


