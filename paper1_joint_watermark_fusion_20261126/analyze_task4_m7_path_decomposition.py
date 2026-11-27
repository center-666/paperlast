"""Path-level attribution of the frozen-table gain over M7-min weights."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import task3
import task4_confirm as confirm
import models_quality as quality
from verify_task4_postrun import independent_replay


HERE = Path(__file__).resolve().parent
OUT = HERE / "task4_confirm"
CONFIG_INDEX = 2
CONFIGURATION = "U_unstable_R35x2"
WATERMARK = "no_repeat"
POLICIES = ("A_M7_min", "S_fixed_table_T")
FIXTURES = ("continuous", "burst")
START, STOP = 50, 110
RECOVERY_STOP = 150


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)),
        "p99": float(np.quantile(values, .99)),
        "p999": float(np.quantile(values, .999)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    folder = parser.parse_args().archives.resolve()
    model = task3.audit_frontend.load_model(CONFIG_INDEX)
    graph = task3.configure(CONFIG_INDEX)[1]
    q_values = np.asarray(quality.frozen.Q, dtype=float)
    request = float(confirm.target_request(task3.read_quality_model(CONFIG_INDEX)))
    request_index = int(np.argmin(abs(q_values-request)))
    phase_rows = []
    path_rows = []
    per_fixture = {}
    traces = []

    for fixture in FIXTURES:
        archive_path = folder / f"{CONFIGURATION}_{fixture}_attack.npz"
        with np.load(archive_path) as archive:
            n = len(archive["path_id"])
            gates = archive[WATERMARK + "_gates"].astype(bool)
            alarm = gates[:, :, 6]
            action_data = {}
            replay_data = {}
            for policy in POLICIES:
                actions = archive[f"{WATERMARK}_{policy}_actions"].astype(int)
                target_q = np.where(alarm, q_values[actions], 1.)
                action_data[policy] = {"actions": actions, "target_q": target_q}
                total_sq, missing_sq, max_missing_sq, violations = independent_replay(
                    model, graph, archive, WATERMARK, policy,
                    start=0, count=n, requested=request)
                stored = archive[f"{WATERMARK}_{policy}_losses"][:, 0]
                replayed = missing_sq[:, START:STOP].mean(axis=1)
                np.testing.assert_allclose(replayed, stored, rtol=3e-8, atol=2e-6)
                replay_data[policy] = {
                    "missing_sq": missing_sq,
                    "stored_losses": stored.copy(),
                    "violations": violations,
                }
                del total_sq, max_missing_sq
            path_id = archive["path_id"].astype(int)
            burst_active = (np.ones(STOP-START, dtype=bool) if fixture == "continuous"
                            else (np.arange(STOP-START) % 10 < 6))
            deltas = (replay_data["S_fixed_table_T"]["missing_sq"][:, START:STOP].mean(axis=1)
                      - replay_data["A_M7_min"]["missing_sq"][:, START:STOP].mean(axis=1))
            q0_fraction = np.divide(
                np.sum(alarm[:, START:STOP] &
                       (action_data["A_M7_min"]["actions"][:, START:STOP] == 0), axis=1),
                np.sum(alarm[:, START:STOP], axis=1),
                out=np.zeros(n), where=np.sum(alarm[:, START:STOP], axis=1) > 0,
            )
            alarm_count = np.sum(alarm[:, START:STOP], axis=1)
            q08_fraction = np.divide(
                np.sum(alarm[:, START:STOP] &
                       (action_data["S_fixed_table_T"]["actions"][:, START:STOP] == request_index), axis=1),
                alarm_count, out=np.zeros(n), where=alarm_count > 0,
            )
            disagreement = np.sum(
                action_data["A_M7_min"]["actions"][:, START:STOP]
                != action_data["S_fixed_table_T"]["actions"][:, START:STOP], axis=1)
            # Show one actual near-median path and the largest-gain path per input.
            for label, i in (("median", np.argmin(abs(deltas-np.median(deltas)))),
                             ("largest", np.argmin(deltas))):
                minimum, table = POLICIES
                errors = [replay_data[p]["missing_sq"][i] for p in POLICIES]
                cumulative = np.cumsum((errors[0]-errors[1])[START:STOP]) / (STOP-START)
                for t in range(1, RECOVERY_STOP):
                    traces.append(dict(fixture=fixture, selection=label, path_id=int(path_id[i]), tick=t,
                                       min_q=action_data[minimum]["target_q"][i, t],
                                       table_q=action_data[table]["target_q"][i, t],
                                       min_error=errors[0][t], table_error=errors[1][t],
                                       cumulative_gain=(cumulative[min(t, STOP-1)-START] if t >= START else 0.)))
            for row in range(n):
                item = {
                    "fixture": fixture,
                    "path_id": int(path_id[row]),
                    "target_alarm_ticks": int(alarm_count[row]),
                    "A_M7_min_attack_mse": float(replay_data["A_M7_min"]["stored_losses"][row]),
                    "T_attack_mse": float(replay_data["S_fixed_table_T"]["stored_losses"][row]),
                    "T_minus_A_M7_min": float(deltas[row]),
                    "equal_weighted_mean_contribution": float(deltas[row]/(2*n)),
                    "A_M7_min_q0_alarm_fraction": float(q0_fraction[row]),
                    "T_q08_alarm_fraction": float(q08_fraction[row]),
                    "action_disagreement_ticks": int(disagreement[row]),
                }
                path_rows.append(item)

            # The burst active/inactive masks alternate every ten ticks. Compute
            # exact phase means separately rather than approximating contiguous spans.
            for policy in POLICIES:
                errors = replay_data[policy]["missing_sq"]
                phases = (("full", "active", "recovery") if fixture == "continuous"
                          else ("full", "active", "inactive", "recovery"))
                for phase in phases:
                    if phase == "full":
                        values = errors[:, START:STOP].mean(axis=1)
                    elif phase == "recovery":
                        values = errors[:, STOP:RECOVERY_STOP].mean(axis=1)
                    else:
                        mask = burst_active if phase == "active" else ~burst_active
                        values = errors[:, START:STOP][:, mask].mean(axis=1)
                    phase_rows.append({
                        "fixture": fixture, "policy": policy, "phase": phase,
                        **distribution(values), "paths": n,
                    })

            tick_rows = []
            for tick in range(1, RECOVERY_STOP):
                values = {}
                for policy in POLICIES:
                    values[policy] = replay_data[policy]["missing_sq"][:, tick]
                if START <= tick < STOP:
                    phase = "active" if burst_active[tick-START] else "inactive"
                elif STOP <= tick < RECOVERY_STOP:
                    phase = "recovery"
                else:
                    phase = "outside_scored_windows"
                row = {
                    "fixture": fixture,
                    "tick": tick,
                    "phase": phase,
                    "target_alarm_fraction": float(alarm[:, tick].mean()),
                    "mean_q_A_M7_min": float(action_data["A_M7_min"]["target_q"][:, tick].mean()),
                    "mean_q_T": float(action_data["S_fixed_table_T"]["target_q"][:, tick].mean()),
                    "mean_error_A_M7_min": float(values["A_M7_min"].mean()),
                    "mean_error_T": float(values["S_fixed_table_T"].mean()),
                    "mean_error_difference_T_minus_M7": float(
                        values["S_fixed_table_T"].mean()-values["A_M7_min"].mean()),
                }
                tick_rows.append(row)
            cumulative = 0.
            for row in tick_rows:
                if START <= row["tick"] < STOP:
                    cumulative += row["mean_error_difference_T_minus_M7"]
                    row["cumulative_window_mse_difference"] = cumulative/(STOP-START)
                else:
                    row["cumulative_window_mse_difference"] = None
            per_fixture[fixture] = {
                "paths": n,
                "alarm_decisions": int(alarm[:, START:STOP].sum()),
                "A_M7_min_action_counts_by_q": {
                    str(value): int(np.sum(alarm[:, START:STOP] &
                                           (action_data["A_M7_min"]["actions"][:, START:STOP] == index)))
                    for index, value in enumerate(q_values)
                },
                "T_action_counts_by_q": {
                    str(value): int(np.sum(alarm[:, START:STOP] &
                                           (action_data["S_fixed_table_T"]["actions"][:, START:STOP] == index)))
                    for index, value in enumerate(q_values)
                },
                "A_M7_min_q0_alarm_decisions": int(np.sum(
                    alarm[:, START:STOP] &
                    (action_data["A_M7_min"]["actions"][:, START:STOP] == 0))),
                "A_M7_min_q_nonzero_alarm_decisions": int(np.sum(
                    alarm[:, START:STOP] &
                    (action_data["A_M7_min"]["actions"][:, START:STOP] != 0))),
                "T_q_requested_alarm_decisions": int(np.sum(
                    alarm[:, START:STOP] &
                    (action_data["S_fixed_table_T"]["actions"][:, START:STOP] == request_index))),
                "A_M7_min_replay_M7_violations": replay_data["A_M7_min"]["violations"],
                "T_replay_M7_violations": replay_data["S_fixed_table_T"]["violations"],
                "path_delta": distribution(deltas),
                "T_better_path_fraction": float(np.mean(deltas < 0)),
                "T_worse_path_fraction": float(np.mean(deltas > 0)),
                "burst_phase_means": ({
                    "A_M7_min_active": float(replay_data["A_M7_min"]["missing_sq"][:, START:STOP][:, burst_active].mean()),
                    "T_active": float(replay_data["S_fixed_table_T"]["missing_sq"][:, START:STOP][:, burst_active].mean()),
                    "A_M7_min_inactive": float(replay_data["A_M7_min"]["missing_sq"][:, START:STOP][:, ~burst_active].mean()),
                    "T_inactive": float(replay_data["S_fixed_table_T"]["missing_sq"][:, START:STOP][:, ~burst_active].mean()),
                } if fixture == "burst" else None),
                "time_series": tick_rows,
            }

    all_deltas = np.asarray([row["T_minus_A_M7_min"] for row in path_rows])
    expected = None
    with (OUT / "paired_direction_corrected.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["group"] == "U_unstable_R35x2_primary"
                    and row["comparison"] == "no_repeat_S_minus_A_M7"
                    and row["metric"] == "attack_full"):
                expected = float(row["difference"])
                break
    if expected is None:
        raise ValueError("Registered paired M7 contrast is missing.")
    if abs(float(all_deltas.mean())-expected) > 2e-8:
        raise ValueError("Path-level means do not reconstruct the registered contrast.")

    benefits = np.sort(all_deltas[all_deltas < 0])
    gross_benefit = float(-benefits.sum())
    gross_harm = float(all_deltas[all_deltas > 0].sum())
    top_benefit = {}
    for percent in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(len(all_deltas)*percent)))
        largest = np.sort(all_deltas)[:count]
        top_benefit[f"top_{int(percent*100)}pct"] = {
            "paths": count,
            "sum_T_minus_A_M7_min": float(largest.sum()),
            "share_of_gross_benefit": float(-largest.sum()/gross_benefit),
            "share_of_net_mean_improvement": float(-largest.sum()/(-all_deltas.sum())),
        }
    q_bins = []
    boundaries = ((0., .5, "below_50pct"), (.5, .8, "50_to_80pct"),
                  (.8, .95, "80_to_95pct"), (.95, 1.000001, "95_to_100pct"))
    for low, high, label in boundaries:
        rows = [row for row in path_rows
                if low <= row["A_M7_min_q0_alarm_fraction"] < high]
        if rows:
            diffs = np.array([row["T_minus_A_M7_min"] for row in rows])
            q_bins.append({
                "bin": label, "paths": len(rows),
                "mean_A_M7_min_q0_fraction": float(np.mean([
                    row["A_M7_min_q0_alarm_fraction"] for row in rows])),
                "mean_A_M7_min_attack_mse": float(np.mean([
                    row["A_M7_min_attack_mse"] for row in rows])),
                "mean_T_attack_mse": float(np.mean([
                    row["T_attack_mse"] for row in rows])),
                "mean_T_minus_A_M7_min": float(diffs.mean()),
                "T_better_path_fraction": float(np.mean(diffs < 0)),
            })

    path_out = OUT / "M7_PATH_CONTRIBUTIONS.csv"
    phase_out = OUT / "M7_PATH_PHASES.csv"
    time_out = OUT / "M7_PATH_TIME_SERIES.csv"
    summary_rows = []
    time_rows = []
    for fixture in FIXTURES:
        summary = per_fixture[fixture]
        for policy in POLICIES:
            phase = [row for row in phase_rows if row["fixture"] == fixture and row["policy"] == policy]
            summary_rows.extend(phase)
        time_rows.extend(summary["time_series"])
    write_csv(path_out, path_rows)
    write_csv(phase_out, summary_rows)
    write_csv(time_out, time_rows)
    trace_out = OUT / "M7_EXAMPLE_TRACES.csv"
    write_csv(trace_out, traces)
    report = {
        "status": "PASS_path_level_reconstruction",
        "configuration": CONFIGURATION,
        "watermark": WATERMARK,
        "policies": list(POLICIES),
        "paths": len(path_rows),
        "windows": {"attack": [START, STOP], "recovery": [STOP, RECOVERY_STOP]},
        "mean_path_T_minus_A_M7_min": float(all_deltas.mean()),
        "registered_paired_difference": expected,
        "difference_reconstruction_abs_error": abs(float(all_deltas.mean())-expected),
        "paired_path_difference": distribution(all_deltas),
        "paired_gain_positive_for_T": distribution(-all_deltas),
        "T_better_path_fraction": float(np.mean(all_deltas < 0)),
        "T_worse_path_fraction": float(np.mean(all_deltas > 0)),
        "gross_benefit_sum": gross_benefit,
        "gross_harm_sum": gross_harm,
        "top_benefit_contribution": top_benefit,
        "q0_fraction_bins": q_bins,
        "fixtures": {
            fixture: {key: value for key, value in summary.items()
                      if key != "time_series"}
            for fixture, summary in per_fixture.items()
        },
        "input_sha256": {
            f"{CONFIGURATION}_{fixture}_attack.npz": sha(
                folder / f"{CONFIGURATION}_{fixture}_attack.npz")
            for fixture in FIXTURES
        },
        "outputs_sha256": {
            path_out.name: sha(path_out), phase_out.name: sha(phase_out),
            time_out.name: sha(time_out), trace_out.name: sha(trace_out),
        },
        "interpretation_scope": "Descriptive reanalysis of the frozen confirmation paths. Path-level contrasts reuse all same-path policy losses and do not add Monte Carlo paths.",
    }
    (OUT / "M7_PATH_DECOMPOSITION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "status": report["status"],
        "paths": report["paths"],
        "mean_difference": report["mean_path_T_minus_A_M7_min"],
        "expected_difference": expected,
        "T_better_path_fraction": report["T_better_path_fraction"],
        "top_1pct": top_benefit["top_1pct"],
    }, indent=2))


if __name__ == "__main__":
    main()
