"""Lock the selected T3 soft policy and all confirmation path definitions."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

import task4_confirm as confirm
import task3

HERE = Path(__file__).resolve().parent
DEV = HERE/"task3_dev"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    destination = HERE/"TASK4_CONFIRM_LOCK.json"
    if destination.exists():
        raise FileExistsError("Task4 lock already exists.")
    run_path = DEV/"RUN.json"
    analysis_path = DEV/"ANALYSIS.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if run["status"] != "complete_task3_development":
        raise ValueError("Task3 development must be complete before the lock.")
    if run["identity"]["namespace"] != [20261126, 94017, 83]:
        raise ValueError("Task4 lock accepts only corrected task3 root 83.")
    if analysis["run_sha256"] != sha(run_path):
        raise ValueError("Task3 analysis is not bound to the current run.")
    for filename, expected in run["archives"].items():
        if sha(DEV/filename) != expected:
            raise ValueError(f"Task3 archive SHA mismatch: {filename}")

    primary_name = "U_unstable_R35x2"
    paired = read_csv(DEV/"paired.csv")
    h_minus_t = next(row for row in paired
                     if row["configuration"] == primary_name
                     and row["comparison"] == "H_fixed_epsilon-T_fixed_table"
                     and row["metric"] == "attack_full")
    d1_minus_d0 = next(row for row in paired
                       if row["configuration"] == primary_name
                       and row["comparison"] == "D1_match_score-D0_duration"
                       and row["metric"] == "path_action_disagreement_fraction")
    if float(d1_minus_d0["difference"]) != 0:
        raise ValueError("The registered primary D1/D0 action difference is nonzero.")

    large_event_thresholds = {}
    t3_archive_hashes = {}
    for config_index, name in enumerate(task3.QUALITY_ORDER):
        archive_path = DEV/f"config{config_index:02}.npz"
        t3_archive_hashes[archive_path.name] = sha(archive_path)
        large_event_thresholds[name] = {}
        with np.load(archive_path) as archive:
            for fixture in confirm.FIXTURES:
                development_mse = archive[f"{fixture}_losses"][:, 7, 0]
                large_event_thresholds[name][fixture] = float(np.quantile(development_mse, .99))

    parent_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=HERE, text=True).strip()
    lock = {
        "status": "locked_before_confirmatory_sampling",
        "lock_creation_parent_commit": parent_commit,
        "selection": "S_fixed_table_T",
        "selection_reason": {
            "primary_configuration": primary_name,
            "H_fixed_epsilon_minus_T_difference": float(h_minus_t["difference"]),
            "H_fixed_epsilon_minus_T_CI95": [float(h_minus_t["lower95"]),
                                              float(h_minus_t["upper95"])],
            "D1_minus_D0_primary_action_disagreement": float(d1_minus_d0["difference"]),
            "decision": "S is the source-condition fixed table T. H increased primary attack and recovery loss. D1 and D0 selected identical primary actions and trajectories.",
        },
        "task3_run_sha256": sha(run_path),
        "task3_analysis_sha256": sha(analysis_path),
        "task3_archive_sha256": t3_archive_hashes,
        "seed_namespace": list(confirm.ROOT_SEED),
        "attack_paths_per_configuration_fixture": confirm.N,
        "healthy_paths_per_configuration": confirm.HEALTH_N,
        "source_hashes": confirm.source_hashes(),
        "large_event_threshold_definition": "For each configuration and fixture, task3 development q1-full-accept node-26 missing-block attack MSE 99th percentile; confirmation event is strict exceedance.",
        "large_event_thresholds": large_event_thresholds,
        "acceptance_threshold": None,
        "acceptance_status": "Engineering MSE tolerances were not provided; report paired effects and intervals without binary acceptance.",
        "windows": {"attack": [50, 110], "recovery": [110, 150], "health": [20, 150]},
        "watermarks": list(confirm.WATERMARKS),
        "policies": list(confirm.POLICIES),
        "configuration_order": list(task3.QUALITY_ORDER),
        "independent_units": "Complete physical trajectories. Four 2x2 cells share same path strata.",
    }
    confirm.quality.frozen.atomic_json(destination, lock)
    print(json.dumps({
        "status": lock["status"], "lock_sha256": sha(destination),
        "namespace": lock["seed_namespace"],
        "event_thresholds": sum(len(x) for x in large_event_thresholds.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
