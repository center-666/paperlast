"""Supplementary readout from frozen W5 arrays, with common path resampling."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


HERE = Path(__file__).resolve().parent
ARMS = (
    "quarter_iid_plugin", "quarter_iid_exact",
    "quarter_no_repeat_plugin", "quarter_no_repeat_exact",
    "negative_plugin", "negative_exact",
)
GROUPS = {
    "all": tuple(range(12)),
    "unstable": tuple(range(6)),
    "stable": tuple(range(6, 12)),
    "onset50": tuple(range(0, 12, 2)),
    "onset65": tuple(range(1, 12, 2)),
}
PAIRS = (
    ("quarter_no_repeat_exact", "quarter_iid_exact", "no_repeat_minus_iid_exact"),
    ("quarter_no_repeat_exact", "negative_exact", "no_repeat_minus_negative_exact"),
    ("quarter_iid_exact", "negative_exact", "iid_minus_negative_exact"),
    ("quarter_no_repeat_plugin", "quarter_iid_plugin", "no_repeat_minus_iid_plugin"),
)
REPS = 10000
SEED = [20261127, 52387, 16, 9913]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cp(count, size):
    return [
        0.0 if count == 0 else float(beta.ppf(.025, count, size-count+1)),
        1.0 if count == size else float(beta.ppf(.975, count+1, size-count)),
    ]


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    folder = HERE / "markov_confirm"
    run = json.loads((folder / "RUN.json").read_text(encoding="utf-8"))
    for name, expected in run["archives"].items():
        assert sha(folder / name) == expected, name

    levels = (.025, .05, .075)
    thresholds = {}
    health_rows = []
    health_values = []
    for model in ("unstable", "stable"):
        with np.load(folder / f"{model}_calibration.npz") as cal:
            for arm in ARMS:
                maxima = np.sort(cal["max_" + arm])
                for level in levels:
                    rank = int(np.ceil((len(maxima)+1) * (1-level)))
                    cutoff = float(maxima[rank-1])
                    thresholds[model, arm, level] = cutoff
                assert thresholds[model, arm, .05] == run["thresholds"][model][arm]["threshold"]
        with np.load(folder / f"{model}_health_0.npz") as data:
            values = []
            for arm in ARMS:
                values.append(np.column_stack([
                    data[arm + "_post20_affected_mse"],
                    data[arm + "_post20_network_mse"],
                ]))
                for level in levels:
                    count = int(np.count_nonzero(data[arm + "_max"] > thresholds[model, arm, level]))
                    size = len(data[arm + "_max"])
                    lo, hi = cp(count, size)
                    health_rows.append({
                        "model": model, "arm": arm, "target_fwer": level,
                        "threshold": thresholds[model, arm, level],
                        "alarms": count, "paths": size, "fwer": count/size,
                        "cp95_lower": lo, "cp95_upper": hi,
                    })
            health_values.append(np.stack(values, axis=1))

    metrics = (
        "early025", "early050", "early075", "attack_mse", "recovery_mse",
        "attack_network_mse", "recovery_network_mse",
    )
    cells = []
    tails = []
    for model in ("unstable", "stable"):
        for scene in range(6):
            with np.load(folder / f"{model}_attack_{scene}.npz") as data:
                by_arm = []
                negative = data["negative_exact_attack_mse"]
                for arm in ARMS:
                    early = data[arm + "_early_max"]
                    np.testing.assert_array_equal(
                        early > thresholds[model, arm, .05],
                        data[arm + "_early_link"].astype(bool),
                    )
                    values = [
                        (early > thresholds[model, arm, level]).mean(axis=1)
                        for level in levels
                    ]
                    values.extend(data[arm + "_" + metric] for metric in metrics[3:])
                    by_arm.append(np.column_stack(values))
                    mse = data[arm + "_attack_mse"]
                    event = (mse > 1) & (mse >= 10*negative)
                    early_failure = ~data[arm + "_early_link"].all(axis=1).astype(bool)
                    for label, flags in (
                        ("registered_tail", event),
                        ("attack_mse_gt1", mse > 1),
                        ("any_edge_early_miss", early_failure),
                    ):
                        count = int(flags.sum())
                        size = len(flags)
                        lo, hi = cp(count, size)
                        tails.append({
                            "model": model, "scene": scene, "arm": arm,
                            "event": label, "count": count, "paths": size,
                            "probability": count/size, "cp95_lower": lo, "cp95_upper": hi,
                            "attack_mse_median": float(np.median(mse)),
                            "attack_mse_p99": float(np.quantile(mse, .99)),
                            "attack_mse_max": float(mse.max()),
                        })
                cells.append(np.stack(by_arm, axis=1))
    cells = np.stack(cells)
    n = cells.shape[1]
    rng = np.random.default_rng(SEED)
    shape = (REPS, len(ARMS), len(metrics))
    bootstrap = {group: np.zeros(shape) for group in GROUPS}
    for scene, values in enumerate(cells):
        weights = rng.multinomial(n, np.full(n, 1/n), size=REPS).astype(float)/n
        draw = (weights @ values.reshape(n, -1)).reshape(shape)
        for group, indices in GROUPS.items():
            if scene in indices:
                bootstrap[group] += draw/len(indices)

    absolute_rows = []
    contrast_rows = []
    for group, indices in GROUPS.items():
        means = cells[list(indices)].mean(axis=(0, 1))
        lower, upper = np.quantile(bootstrap[group], [.025, .975], axis=0)
        for i, arm in enumerate(ARMS):
            for j, metric in enumerate(metrics):
                absolute_rows.append({
                    "group": group, "arm": arm, "metric": metric,
                    "mean": means[i, j], "lower": lower[i, j], "upper": upper[i, j],
                })
        for left, right, label in PAIRS:
            a, b = ARMS.index(left), ARMS.index(right)
            diff = bootstrap[group][:, a] - bootstrap[group][:, b]
            lo, hi = np.quantile(diff, [.025, .975], axis=0)
            for j, metric in enumerate(metrics):
                contrast_rows.append({
                    "group": group, "comparison": label, "metric": metric,
                    "difference": means[a, j]-means[b, j], "lower": lo[j], "upper": hi[j],
                })

    health_contrasts = []
    for model, values in zip(("unstable", "stable"), health_values):
        size = values.shape[0]
        weights = rng.multinomial(size, np.full(size, 1/size), size=REPS).astype(float)/size
        draw = (weights @ values.reshape(size, -1)).reshape(REPS, len(ARMS), 2)
        means = values.mean(axis=0)
        for left, right, label in PAIRS:
            a, b = ARMS.index(left), ARMS.index(right)
            lo, hi = np.quantile(draw[:, a]-draw[:, b], [.025, .975], axis=0)
            for j, metric in enumerate(("post20_affected_mse", "post20_network_mse")):
                health_contrasts.append({
                    "model": model, "comparison": label, "metric": metric,
                    "left_mean": means[a, j], "right_mean": means[b, j],
                    "difference": means[a, j]-means[b, j],
                    "lower": lo[j], "upper": hi[j],
                    "relative_mean_difference": means[a, j]/means[b, j]-1,
                })

    zero_bounds = []
    for group, indices in GROUPS.items():
        selected = [tails[scene*len(ARMS)*3 + arm*3 + event]
                    for scene in indices for arm in range(len(ARMS)) for event in range(3)]
        for arm in ARMS:
            for event in ("registered_tail", "attack_mse_gt1", "any_edge_early_miss"):
                records = [r for r in selected if r["arm"] == arm and r["event"] == event]
                count = sum(r["count"] for r in records)
                size = sum(r["paths"] for r in records)
                record = {"group": group, "arm": arm, "event": event,
                          "count": count, "paths": size, "probability": count/size}
                if count == 0:
                    record["zero_event_mean_probability_upper95_one_sided"] = float(
                        -np.expm1(np.log(.05)/size))
                zero_bounds.append(record)

    save_csv(folder / "sensitivity_health.csv", health_rows)
    save_csv(folder / "supplement_absolute.csv", absolute_rows)
    save_csv(folder / "supplement_paired.csv", contrast_rows)
    save_csv(folder / "supplement_health_paired.csv", health_contrasts)
    save_csv(folder / "supplement_scene_events.csv", tails)
    summary = {
        "run_sha256": sha(folder / "RUN.json"),
        "source_sha256": sha(Path(__file__)),
        "bootstrap_replicates": REPS, "bootstrap_seed": SEED,
        "resampling": "One path weight per fixed scene shared across all arms and metrics. Healthy paths resampled independently within model with shared arm weights.",
        "interval_scope": "Pointwise paired 95% intervals conditional on frozen thresholds. No family-wise correction.",
        "sensitivity_scope": "Only early scores and healthy maxima are re-thresholded at 2.5/5/7.5%. Closed-loop MSE retains the locked 5% gates.",
        "zero_event_bound": "For independent equal-size strata, P(no events) = product_s(1-p_s)^n <= (1-mean_s p_s)^(n*S), by concavity of log(1-p). The inverted one-sided 95% upper bound is valid for the equal-stratum mean probability, including heterogeneous p_s.",
        "event_counts": zero_bounds,
        "resource_wall_seconds_all_arms_8_workers": run["wall_seconds"],
        "outputs": {},
    }
    for filename in (
        "sensitivity_health.csv", "supplement_absolute.csv", "supplement_paired.csv",
        "supplement_health_paired.csv", "supplement_scene_events.csv",
    ):
        summary["outputs"][filename] = sha(folder / filename)
    (folder / "SUPPLEMENT_READOUT.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"paths": 12*n, "verified_archives": len(run["archives"]),
                      "output_rows": [len(absolute_rows), len(contrast_rows), len(tails)]}))


if __name__ == "__main__":
    summarize()
