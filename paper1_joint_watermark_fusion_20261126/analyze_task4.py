"""Path-paired, configuration-stratified analysis of the frozen 2x2 confirmation."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import models_quality as quality
import task4_confirm as confirm

HERE = Path(__file__).resolve().parent
OUT = HERE/"task4_confirm"
REPS = 20000
SEED = [20261126, 94017, 84, 9911]
WATERMARKS = confirm.WATERMARKS
POLICIES = confirm.POLICIES
CELLS = tuple((wm, policy) for wm in WATERMARKS for policy in POLICIES)
METRICS = ("attack_full", "attack_active", "attack_inactive", "network_attack",
           "recovery", "health_post20", "network_post20", "max_missing_sq")
def stratum_indices(configuration_indices):
    return tuple(2*config+fixture for config in configuration_indices for fixture in (0, 1))


GROUPS = {
    "all_12_configurations": tuple(range(24)),
    "U_all_dynamics_quality": stratum_indices((0, 1, 2, 6, 7, 8)),
    "U_unstable_all_quality": stratum_indices((0, 1, 2)),
    "U_stable_all_quality": stratum_indices((6, 7, 8)),
    "R_all_dynamics_quality": stratum_indices((3, 4, 5, 9, 10, 11)),
    "U_unstable_R35x2_primary": (4, 5),
}
PAIRS = (
    ("iid", "A_q0_raw", "iid", "S_fixed_table_T", "iid_S_minus_A"),
    ("no_repeat", "A_q0_raw", "no_repeat", "S_fixed_table_T", "no_repeat_S_minus_A"),
    ("iid", "A_M7_min", "iid", "S_fixed_table_T", "iid_S_minus_A_M7"),
    ("no_repeat", "A_M7_min", "no_repeat", "S_fixed_table_T", "no_repeat_S_minus_A_M7"),
    ("iid", "A_q0_raw", "no_repeat", "A_q0_raw", "no_repeat_minus_iid_A"),
    ("iid", "S_fixed_table_T", "no_repeat", "S_fixed_table_T", "no_repeat_minus_iid_S"),
)


def csvout(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cp(count, n):
    low = 0. if count == 0 else float(beta.ppf(.025, count, n-count+1))
    high = 1. if count == n else float(beta.ppf(.975, count+1, n-count))
    return low, high


def main():
    run = quality.json.loads((OUT/"RUN.json").read_text(encoding="utf-8"))
    lock = quality.json.loads((HERE/"TASK4_CONFIRM_LOCK.json").read_text(encoding="utf-8"))
    for filename, expected in run["archives"].items():
        assert confirm.sha(OUT/filename) == expected
    rng = np.random.default_rng(SEED)
    config_records = []
    strata = []
    absolute_rows = []
    fwer_rows = []
    scene_rows = []
    event_rows = []
    for config_index, name in enumerate(confirm.QUALITY_ORDER):
        record = run["configs"][name]
        thresholds = record["thresholds"]
        event_thresholds = lock["large_event_thresholds"][name]
        for fixture in confirm.FIXTURES:
            path = OUT/f"{name}_{fixture}_attack.npz"
            with np.load(path) as archive:
                n = len(archive["path_id"])
                assert n == confirm.N
                losses = np.stack([
                    archive[wm+"_"+pol+"_losses"]
                    for wm, pol in CELLS
                ], axis=1)
                strata.append({
                    "config": config_index, "fixture": fixture,
                    "name": name, "losses": losses, "n": n,
                })
                for c_index, (wm, policy) in enumerate(CELLS):
                    for metric_index, metric in enumerate(METRICS):
                        values = losses[:, c_index, metric_index]
                        absolute_rows.append({
                            "configuration": name, "fixture": fixture,
                            "watermark": wm, "fusion": policy, "metric": metric,
                            "mean": float(values.mean()), "median": float(np.median(values)),
                            "p99": float(np.quantile(values, .99)), "max": float(values.max()),
                            "n": n,
                        })
                    threshold = float(event_thresholds[fixture])
                    events = losses[:, c_index, 0] > threshold
                    k = int(events.sum())
                    low, high = cp(k, n)
                    event_rows.append({
                        "configuration": name, "fixture": fixture,
                        "watermark": wm, "fusion": policy,
                        "development_q1_p99_threshold": threshold,
                        "count": k, "paths": n, "probability": k/n,
                        "CP95_lower": low, "CP95_upper": high,
                    })
                    scene_rows.append({
                        "configuration": name, "fixture": fixture,
                        "watermark": wm, "fusion": policy,
                        "target_alarm_paths": int(archive[wm+"_gates"][:, :, 6].astype(bool).any(axis=1).sum()),
                        "target_alarm_decisions": int(archive[wm+"_gates"][:, :, 6].sum()),
                        "M7_violations": int(archive[wm+"_"+policy+"_M7_violations"]),
                        "M7_projection_count": int(archive[wm+"_"+policy+"_M7_fallbacks"]),
                    })
        health_path = OUT/f"{name}_continuous_health.npz"
        with np.load(health_path) as archive:
            for wm in WATERMARKS:
                gate = archive[wm+"_gates"].astype(bool)
                for family, indicators in (
                    ("target_edge", gate[:, :, 6].any(axis=1)),
                    ("all_seven_edges", gate.any(axis=(1, 2))),
                ):
                    count = int(indicators.sum())
                    n = len(indicators)
                    low, high = cp(count, n)
                    fwer_rows.append({
                        "configuration": name, "watermark": wm,
                        "family": family, "alarms": count, "paths": n,
                        "FWER": count/n, "CP95_lower": low, "CP95_upper": high,
                    })
            config_records.append({"configuration": name, "thresholds": thresholds})

    boot_by_group = {}
    for group, selected_strata in GROUPS.items():
        selected = [strata[i] for i in selected_strata]
        cells = np.stack([item["losses"] for item in selected])  # [stratum,path,cell,metric]
        sample_count = cells.shape[1]
        draw = np.zeros((REPS, len(CELLS), len(METRICS)))
        # A common complete-path weight preserves both watermark rows and all four fusion cells.
        for scene, values in enumerate(cells):
            for first in range(0, REPS, 1000):
                size = min(1000, REPS-first)
                weights = rng.multinomial(sample_count, np.full(sample_count, 1/sample_count),
                                          size=size)/sample_count
                draw[first:first+size] += (
                    weights @ values.reshape(sample_count, -1)
                ).reshape(size, len(CELLS), len(METRICS))/len(cells)
        means = cells.mean(axis=(0, 1))
        lower, upper = np.quantile(draw, (.025, .975), axis=0)
        boot_by_group[group] = {"draws": draw, "means": means, "lower": lower, "upper": upper}
        for c_index, (wm, policy) in enumerate(CELLS):
            for m_index, metric in enumerate(METRICS):
                absolute_rows.append({
                    "configuration": group, "fixture": "equal_weighted",
                    "watermark": wm, "fusion": policy, "metric": metric,
                    "mean": means[c_index, m_index], "median": None,
                    "p99": None, "max": None, "n": sum(item["n"] for item in cells),
                    "lower95": lower[c_index, m_index], "upper95": upper[c_index, m_index],
                })
    paired_rows = []
    for group, result in boot_by_group.items():
        for left_wm, left_policy, right_wm, right_policy, label in PAIRS:
            left = CELLS.index((left_wm, left_policy))
            right = CELLS.index((right_wm, right_policy))
            differences = result["draws"][:, left]-result["draws"][:, right]
            low, high = np.quantile(differences, (.025, .975), axis=0)
            point = result["means"][left]-result["means"][right]
            for metric_index, metric in enumerate(METRICS):
                paired_rows.append({
                    "group": group, "comparison": label, "metric": metric,
                    "difference": point[metric_index], "lower95": low[metric_index],
                    "upper95": high[metric_index],
                })
        # Confirm the interaction between watermark and soft fusion directly.
        old_left = CELLS.index(("iid", "S_fixed_table_T"))
        old_right = CELLS.index(("iid", "A_q0_raw"))
        new_left = CELLS.index(("no_repeat", "S_fixed_table_T"))
        new_right = CELLS.index(("no_repeat", "A_q0_raw"))
        interaction = ((result["draws"][:, new_left]-result["draws"][:, new_right])
                       -(result["draws"][:, old_left]-result["draws"][:, old_right]))
        low, high = np.quantile(interaction, (.025, .975), axis=0)
        point = ((result["means"][new_left]-result["means"][new_right])
                 -(result["means"][old_left]-result["means"][old_right]))
        for metric_index, metric in enumerate(METRICS):
            paired_rows.append({
                "group": group, "comparison": "interaction_new_vs_old_S_minus_A",
                "metric": metric, "difference": point[metric_index],
                "lower95": low[metric_index], "upper95": high[metric_index],
            })

    zero_event_rows = []
    for group, selected_strata in GROUPS.items():
        for c_index, (wm, policy) in enumerate(CELLS):
            records = [event_rows[i] for i in range(len(event_rows))
                       if event_rows[i]["watermark"] == wm
                       and event_rows[i]["fusion"] == policy
                       and any(strata[k]["name"] == event_rows[i]["configuration"]
                               and strata[k]["fixture"] == event_rows[i]["fixture"]
                               for k in selected_strata)]
            count = sum(row["count"] for row in records)
            paths = sum(row["paths"] for row in records)
            item = {
                "group": group, "watermark": wm, "fusion": policy,
                "count": count, "paths": paths, "probability": count/paths,
            }
            if count == 0:
                item["zero_event_upper95_one_sided"] = float(-np.expm1(np.log(.05)/paths))
            else:
                item["zero_event_upper95_one_sided"] = None
            zero_event_rows.append(item)

    csvout(OUT/"absolute.csv", absolute_rows)
    csvout(OUT/"paired.csv", paired_rows)
    csvout(OUT/"scenes.csv", scene_rows)
    csvout(OUT/"health_fwer.csv", fwer_rows)
    csvout(OUT/"large_events.csv", event_rows)
    csvout(OUT/"event_summary.csv", zero_event_rows)
    report = {
        "status": "complete_task4_confirmatory_analysis",
        "run_sha256": quality.sha(OUT/"RUN.json"),
        "lock_sha256": quality.sha(HERE/"TASK4_CONFIRM_LOCK.json"),
        "source_sha256": quality.sha(Path(__file__)),
        "bootstrap_replicates": REPS, "bootstrap_seed": SEED,
        "unit": "Independent physical path within fixed configuration and fixture, paired over all four 2x2 cells.",
        "groups": {name: list(indices) for name, indices in GROUPS.items()},
        "intervals": "Pointwise, paired, path-level percentile bootstrap. Group means weight the configured scenario-fixture strata equally.",
        "outputs_sha256": {f: quality.sha(OUT/f) for f in (
            "absolute.csv", "paired.csv", "scenes.csv", "health_fwer.csv",
            "large_events.csv", "event_summary.csv")},
    }
    quality.frozen.atomic_json(OUT/"ANALYSIS.json", report)
    print(json.dumps({"status": report["status"], "paths": len(strata)*confirm.N,
                      "paired_rows": len(paired_rows), "health_fwer_rows": len(fwer_rows)}))


if __name__ == "__main__":
    main()
