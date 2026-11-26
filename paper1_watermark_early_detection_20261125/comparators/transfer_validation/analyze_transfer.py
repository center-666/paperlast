"""Scene-stratified whole-path paired statistics for the frozen transfer study."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import study_transfer as study

HERE = Path(__file__).resolve().parent
REPLICATES = 20000
GROUPS = {
    "all": tuple(range(12)), "unstable": tuple(range(6)),
    "stable": tuple(range(6, 12)), "onset50": tuple(range(0, 12, 2)),
    "onset65": tuple(range(1, 12, 2)),
}
CONTRASTS = (
    ("no_repeat_exact", "iid_exact"),
    ("no_repeat_exact", "no_opposite_exact"),
    ("no_opposite_exact", "iid_exact"),
    ("no_repeat_exact", "negative_exact"),
    ("no_repeat_plugin", "iid_plugin"),
    ("no_repeat_plugin", "no_opposite_plugin"),
)
EVENT_METRICS = ("clean_any_early", "exact_early", "exact_full", "compatible_path",
                 "large_late_accept_path", "tail_event", "mse_gt1")


def cp(count, size):
    return (0. if count == 0 else float(beta.ppf(.025, count, size-count+1)),
            1. if count == size else float(beta.ppf(.975, count+1, size-count)))


def csvout(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_health(folder, run, rng):
    rows = []
    sensitivity = []
    for model in study.MODELS:
        with np.load(folder / f"{model}_p0_s0.npz") as cal, \
                np.load(folder / f"{model}_p1_s0.npz") as healthy:
            values = healthy["health_mse"]
            n = len(values)
            boot = np.empty((REPLICATES, len(study.ARMS), 2))
            for start in range(0, REPLICATES, 1000):
                size = min(1000, REPLICATES-start)
                weights = rng.multinomial(n, np.full(n, 1/n), size=size)/n
                boot[start:start+size] = (weights @ values.reshape(n, -1)).reshape(
                    size, len(study.ARMS), 2)
            means = values.mean(axis=0)
            for left, right in CONTRASTS:
                a, b = study.ARMS.index(left), study.ARMS.index(right)
                lo, hi = np.quantile(boot[:, a]-boot[:, b], (.025, .975), axis=0)
                for m, metric in enumerate(("five_mse_post20", "network_mse_post20")):
                    rows.append({"model": model, "comparison": left+"-"+right,
                                 "metric": metric, "left_mean": means[a, m],
                                 "right_mean": means[b, m], "difference": means[a, m]-means[b, m],
                                 "lower": lo[m], "upper": hi[m]})
            for a, arm in enumerate(study.ARMS):
                for alpha in (.025, .05, .075):
                    cutoff = study.cutoff(cal["maxima"][:, a], alpha)
                    count = int((healthy["maxima"][:, a] > cutoff["threshold"]).sum())
                    lo, hi = cp(count, n)
                    sensitivity.append({
                        "model": model, "arm": arm, "target_fwer": alpha,
                        "threshold": cutoff["threshold"], "alarms": count, "paths": n,
                        "fwer": count/n, "lower": lo, "upper": hi,
                        "five_mse_post20_at5": means[a, 0],
                        "network_mse_post20_at5": means[a, 1],
                    })
    csvout(folder / "health_paired.csv", rows)
    csvout(folder / "health_fwer.csv", sensitivity)


def analyze(split):
    folder = HERE / split
    run = json.loads((folder / "RUN.json").read_text(encoding="utf-8"))
    for filename, digest in run["archives"].items():
        assert study.sha(folder / filename) == digest
    seed = [20261126, 61783, run["identity"]["split"], 9001]
    rng = np.random.default_rng(seed)
    cells = []
    scene_rows = []
    event_rows = []
    sensitivity_rows = []
    for scene_index, (model, angle, onset) in enumerate(run["identity"]["scenes"]):
        scene = scene_index % 6
        with np.load(folder / f"{model}_p2_s{scene}.npz") as data:
            values = data["values"]
            cells.append(values)
            for s, (support, edges) in enumerate(study.SUPPORTS):
                for a, arm in enumerate(study.ARMS):
                    for m, metric in enumerate(study.METRICS):
                        scene_rows.append({
                            "model": model, "angle": angle, "onset": onset,
                            "support": support, "arm": arm, "metric": metric,
                            "mean": float(values[:, s, a, m].mean()),
                        })
                    mse = values[:, s, a, study.METRICS.index("attack_five_mse")]
                    for metric in EVENT_METRICS:
                        count = int(values[:, s, a, study.METRICS.index(metric)].sum())
                        n = len(values)
                        lo, hi = cp(count, n)
                        event_rows.append({
                            "model": model, "angle": angle, "onset": onset,
                            "support": support, "arm": arm, "event": metric,
                            "count": count, "paths": n, "probability": count/n,
                            "cp95_lower": lo, "cp95_upper": hi,
                            "mse_median": float(np.median(mse)),
                            "mse_p99": float(np.quantile(mse, .99)),
                            "mse_max": float(mse.max()),
                        })
            with np.load(folder / f"{model}_p0_s0.npz") as cal:
                for alpha in (.025, .05, .075):
                    thresholds = np.array([study.cutoff(cal["maxima"][:, a], alpha)["threshold"]
                                           for a in range(len(study.ARMS))])
                    early = data["early_max"] > thresholds[None, None, :, None]
                    for s, (support, edges) in enumerate(study.SUPPORTS):
                        clean = np.setdiff1d(np.arange(6), edges)
                        for a, arm in enumerate(study.ARMS):
                            attacked = early[:, s, a][:, edges].mean(axis=1)
                            clean_any = early[:, s, a][:, clean].any(axis=1)
                            exact = early[:, s, a][:, edges].all(axis=1) & ~clean_any
                            sensitivity_rows.append({
                                "model": model, "angle": angle, "onset": onset,
                                "support": support, "arm": arm, "target_fwer": alpha,
                                "attacked_early": float(attacked.mean()),
                                "exact_early": float(exact.mean()),
                                "clean_any_early": float(clean_any.mean()),
                            })
    cells = np.stack(cells)
    n = cells.shape[1]
    shape = (REPLICATES, *cells.shape[2:])
    boot = {group: np.zeros(shape) for group in GROUPS}
    for scene, values in enumerate(cells):
        # Reuse these physical-path weights for every support, arm and outcome.
        for start in range(0, REPLICATES, 1000):
            size = min(1000, REPLICATES-start)
            weights = rng.multinomial(n, np.full(n, 1/n), size=size)/n
            sample = (weights @ values.reshape(n, -1)).reshape(size, *cells.shape[2:])
            for group, indices in GROUPS.items():
                if scene in indices:
                    boot[group][start:start+size] += sample/len(indices)
    absolute_rows = []
    contrast_rows = []
    counts = []
    for group, indices in GROUPS.items():
        point = cells[list(indices)].mean(axis=(0, 1))
        lo, hi = np.quantile(boot[group], (.025, .975), axis=0)
        for s, (support, _) in enumerate(study.SUPPORTS):
            for a, arm in enumerate(study.ARMS):
                for m, metric in enumerate(study.METRICS):
                    absolute_rows.append({
                        "group": group, "support": support, "arm": arm, "metric": metric,
                        "mean": point[s, a, m], "lower": lo[s, a, m], "upper": hi[s, a, m],
                    })
                for metric in EVENT_METRICS:
                    m = study.METRICS.index(metric)
                    flags = cells[list(indices), :, s, a, m]
                    count = int(flags.sum())
                    size = flags.size
                    counts.append({
                        "group": group, "support": support, "arm": arm, "event": metric,
                        "count": count, "paths": size, "probability": count/size,
                        "zero_event_upper95_one_sided": float(-np.expm1(np.log(.05)/size))
                        if count == 0 else None,
                    })
            for left, right in CONTRASTS:
                a, b = study.ARMS.index(left), study.ARMS.index(right)
                difference = boot[group][:, s, a] - boot[group][:, s, b]
                lower, upper = np.quantile(difference, (.025, .975), axis=0)
                family8_lo, family8_hi = np.quantile(difference, (.025/8, 1-.025/8), axis=0)
                family3_lo, family3_hi = np.quantile(difference, (.025/3, 1-.025/3), axis=0)
                for m, metric in enumerate(study.METRICS):
                    contrast_rows.append({
                        "group": group, "support": support, "comparison": left+"-"+right,
                        "metric": metric, "difference": point[s, a, m]-point[s, b, m],
                        "lower": lower[m], "upper": upper[m],
                        "bonferroni8_lower": family8_lo[m], "bonferroni8_upper": family8_hi[m],
                        "bonferroni3_lower": family3_lo[m], "bonferroni3_upper": family3_hi[m],
                    })
    del boot
    paired_health(folder, run, rng)
    csvout(folder / "absolute.csv", absolute_rows)
    csvout(folder / "paired.csv", contrast_rows)
    csvout(folder / "scenes.csv", scene_rows)
    csvout(folder / "scene_events.csv", event_rows)
    csvout(folder / "sensitivity.csv", sensitivity_rows)
    summary = {
        "run_sha256": study.sha(folder / "RUN.json"),
        "analysis_source_sha256": study.sha(Path(__file__)),
        "bootstrap_replicates": REPLICATES, "bootstrap_seed": seed,
        "unit": "Independent physical path within fixed scene. Common weights across all supports, methods and outcomes.",
        "intervals": "Conditional on calibrated thresholds. Eight-support exact-set primary family uses Bonferroni8 intervals. Three mechanism event diagnostics use Bonferroni3 intervals. Other intervals are pointwise.",
        "event_counts": counts,
        "table_sha256": {name: study.sha(folder / name) for name in (
            "absolute.csv", "paired.csv", "scenes.csv", "scene_events.csv",
            "sensitivity.csv", "health_paired.csv", "health_fwer.csv")},
    }
    study.atomic_json(folder / "ANALYSIS.json", summary)
    if split == "dev":
        study.atomic_json(HERE / "CONFIRM_LOCK.json", {
            "development_run_sha256": study.sha(folder / "RUN.json"),
            "development_analysis_sha256": study.sha(folder / "ANALYSIS.json"),
            "sources": run["identity"]["sources"],
            "paths_per_scene": study.SPLITS["confirm"][1],
            "health_calibration_and_holdout_each": study.SPLITS["confirm"][2],
            "selection": "Both preregistered transition rules retained, all supports retained, no parameter selection.",
        })
    print(split, "analysis complete", len(absolute_rows), len(contrast_rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("dev", "confirm"))
    analyze(parser.parse_args().split)
