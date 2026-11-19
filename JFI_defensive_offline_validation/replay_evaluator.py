"""Re-run frozen receivers on saved synthetic observations and score paired outputs."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np

from onset_diagnostics import capture_posterior, diagnose_a0
from paired_analysis import defender_comparison, v1a_tradeoff
from risk_metrics import BF_THRESHOLD, DANGER_SQUARED, PathMetrics, events_from_times, score_estimates


HERE = Path(__file__).resolve().parent
BASELINE = "0beb3edace40d514d1969cde205eba3f3b2272be"
MODELS = ("unstable", "stable")
METHODS = ("V0", "V1a", "V1.2a", "V1.4", "anchor@2%")
PROFILES = ("A0", "A1_H_mismatch", "A1_C_mismatch")

REFERENCE = Path("common_g_reference_revision_20261115")
MODEL = REFERENCE / "inputs/five_node_aggregate_model.json"
V1 = Path("JFI_reconstruction_attack_aware_soft_accommodation_V1")
V11 = Path("JFI_V11_soft_weight_reconstruction_20261118")
FRONTIER = Path("JFI_secure_estimation_accuracy_safety_frontier")
OBSERVATIONS = Path("attack_gain_switch_audit_20261115/screen_arrays.npz")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_inputs(base):
    manifest = read_json(HERE / "frozen_inputs_sha256.json")
    if manifest["baseline"] != BASELINE:
        raise ValueError("Unexpected baseline registration.")
    for path, expected in manifest["files"].items():
        data = (base / path).read_bytes()
        if Path(path).suffix in (".py", ".json"):
            data = data.replace(b"\r\n", b"\n")
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return manifest


def load_receiver_modules(base):
    """Resolve the existing estimator and its original observation model."""
    paths = sys.path[:]
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(base / folder) for folder in (FRONTIER, REFERENCE)]
    try:
        receiver = import_module("receiver")
        common = import_module("common_g_reference")
        evidence = import_module("accommodation")
    finally:
        sys.path[:] = paths
    for module, folder in ((receiver, FRONTIER), (common, REFERENCE), (evidence, V1)):
        expected = base / folder / f"{module.__name__}.py"
        if Path(module.__file__).resolve() != expected:
            raise ValueError(f"Wrong frozen module for {module.__name__}.")
    return receiver, common, evidence


def saved_sensor_inputs(base, name, settings, path_ids):
    with np.load(base / OBSERVATIONS, allow_pickle=False) as bank:
        measurements = bank[f"{name}__physical__y"][path_ids]
        decoders = bank[f"{name}__M_target"][path_ids]
    local = measurements[:, :, settings.receiver]
    neighbor = measurements[:, :, settings.healthy]
    trusted = np.concatenate((local, neighbor), axis=2)
    target = measurements[:, :, settings.target]
    for array in (trusted, target, decoders):
        if not np.all(np.isfinite(array)):
            raise ValueError("Nonfinite saved sensor input.")
        array.flags.writeable = False
    return trusted, target, decoders


def compare_replay_with_log(run, trace, saved, profile, legacy_policies):
    checks = {}
    pairs = {
        "log_bf": (run.log_bf, saved[f"{profile}__log_bf"]),
        "observer_log_bf": (trace["log_bf"], run.log_bf),
        "posterior_weights": (trace["weights"], saved[f"{profile}__onset_weights"]),
        "estimated_distortion": (trace["delta"], saved[f"{profile}__delta_hat"]),
    }
    for method in ("V0", "V1a"):
        pairs[method] = (run.post[:, METHODS.index(method)],
                         saved[f"{profile}__post"][:, legacy_policies.index(method)])
    for label, (actual, expected) in pairs.items():
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11,
                                   err_msg=f"{profile}: {label}")
        checks[label] = float(np.max(np.abs(actual - expected)))
    np.testing.assert_array_equal(run.log_bf >= np.log(BF_THRESHOLD),
                                  saved[f"{profile}__common_alarm"])
    return checks


def path_rows(row, path_ids):
    ev = row.events

    def stamp(times, index):
        return int(times[index]) if times[index] < ev.horizon else None

    return [
        {
            "path_id": int(label),
            "first_alarm": stamp(ev.alarm, i),
            "first_danger": stamp(ev.danger, i),
            "strictly_before_alarm": bool(ev.before[i]),
            "first_packet_danger": bool(ev.first_packet[i]),
            "healthy_mse": float(row.healthy_mse[i]),
            "active_mse": float(row.support_mse[i]),
            "after_stop_mse": float(row.tail_mse[i]),
        }
        for i, label in enumerate(path_ids)
    ]


def replay_model(base, name, settings, spec, table, receiver, common, evidence):
    prepared = evidence.PreparedModel(common.load_model(name, base / MODEL), settings)
    observer = receiver.BatchedObserver(prepared)
    path_ids = spec["samples"]["physical_indices"]
    trusted, target, decoder = saved_sensor_inputs(base, name, settings, path_ids)
    legacy = read_json(base / V11 / "phase1_results/ablation.json")["policies"]
    start = np.full(len(path_ids), spec["evaluation_onset"])
    stop = start + settings.template_duration
    if stop.max() >= settings.horizon:
        raise ValueError("A tail segment is required for these registered A0 records.")

    clean_obs = receiver.Observations(trusted, target, decoder)
    clean = receiver.run_receiver(prepared, clean_obs, table)
    clean_trace = capture_posterior(observer, clean_obs)
    cases, checks, detail = {}, {}, {}
    filename = base / V11 / f"phase1_results/{name}_step_logs.npz"
    with np.load(filename, allow_pickle=False) as saved:
        checks["healthy"] = compare_replay_with_log(clean, clean_trace, saved, "healthy", legacy)
        with np.load(base / OBSERVATIONS, allow_pickle=False) as bank:
            truth = bank[f"{name}__physical__x"][path_ids]
        np.testing.assert_array_equal(truth, saved["healthy__state_truth"])
        for profile in PROFILES:
            # Assemble the recorded decoded quantity from its separately saved terms.
            recorded_change = saved[f"{profile}__decoded_difference"]
            received = target + recorded_change
            received.flags.writeable = False
            obs = receiver.Observations(trusted, received, decoder)
            run = receiver.run_receiver(prepared, obs, table)
            trace = capture_posterior(observer, obs)
            checks[profile] = compare_replay_with_log(run, trace, saved, profile, legacy)
            np.testing.assert_array_equal(truth, saved[f"{profile}__state_truth"])
            np.testing.assert_allclose(run.post[:, :, :start[0]], clean.post[:, :, :start[0]],
                                       rtol=0, atol=2e-12)

            metrics = {
                method: score_estimates(run.post[:, j], clean.post[:, j], truth, run.log_bf, start, stop)
                for j, method in enumerate(METHODS)
            }
            cases[profile] = defender_comparison(metrics)
            if profile == "A0":
                cases[profile]["v1a_vs_state_saturation"] = v1a_tradeoff(
                    metrics["V1a"], metrics["anchor@2%"], path_ids
                )
                diagnosis, paths = diagnose_a0(
                    trace, clean_trace, recorded_change, metrics["V1a"],
                    prepared.model.G[settings.receiver], path_ids, settings.template_duration
                )
                cases[profile]["onset_diagnostics"] = diagnosis
                detail["identified_paths"] = paths
                detail["methods"] = {method: path_rows(result, path_ids)
                                     for method, result in metrics.items()}

    return {
        "physical_path_ids": path_ids,
        "source": "Archived decoded receiver inputs and V11 phase-1 reference logs",
        "recorded_onset": int(start[0]),
        "recorded_stop_exclusive": int(stop[0]),
        "healthy_alarm_count": int(np.sum((clean.log_bf >= np.log(BF_THRESHOLD)).any(axis=1))),
        "frozen_log_comparison_max_abs": checks,
        "cases": cases,
    }, detail


def stored_comparison(base, name, columns, horizon, fixed_onset):
    """Audit saved holdout outcomes whose raw input arrays were not retained."""
    cases = {}
    report = (
        ("A0_fixed_onset", "unknown_start"),
        ("A0_all_three_onsets", "unknown_start"),
        ("A1_H_offline_mismatch", "A1_H_replay"),
        ("A1_C_offline_mismatch", "A1_C_replay"),
    )
    with np.load(base / FRONTIER / f"results/{name}_paired.npz", allow_pickle=False) as saved:
        for label, profile in report:
            onset = saved[f"{profile}__onset"]
            stop = saved[f"{profile}__stop"]
            if np.any(stop - onset != 10):
                raise ValueError("The requested holdout profile is not ten ticks.")
            chosen = onset == fixed_onset if label == "A0_fixed_onset" else np.ones(len(onset), dtype=bool)
            alarm = saved[f"{profile}__first_alarm"][chosen]
            selected = np.ix_(chosen, columns)
            danger = saved[f"{profile}__first_danger"][selected]
            first_packet = saved[f"{profile}__first_packet"][selected]
            before = saved[f"{profile}__pre_alarm"][selected]
            active = saved[f"{profile}__active_mean_mse"][selected]
            after = saved[f"{profile}__tail_mean_mse"][selected]
            healthy = saved[f"{profile}__health_mean_mse"][selected]
            np.testing.assert_array_equal(healthy, saved["healthy__health_mean_mse"][selected])
            np.testing.assert_array_equal(danger < horizon,
                                          saved[f"{profile}__peak_paired_squared"][selected] >= DANGER_SQUARED)
            metrics = {}
            for j, method in enumerate(METHODS):
                events = events_from_times(danger[:, j], alarm, onset[chosen], stop[chosen], horizon)
                np.testing.assert_array_equal(events.before, before[:, j])
                np.testing.assert_array_equal(events.first_packet, first_packet[:, j])
                metrics[method] = PathMetrics(events, healthy[:, j], active[:, j], after[:, j])
            cases[label] = defender_comparison(metrics)
            cases[label]["archived_profile"] = profile
            cases[label]["recorded_onsets"] = np.unique(onset[chosen]).tolist()
            if label.startswith("A0"):
                cases[label]["v1a_vs_state_saturation"] = v1a_tradeoff(
                    metrics["V1a"], metrics["anchor@2%"], np.flatnonzero(chosen)
                )
    return cases


def evaluate(base):
    base = Path(base).resolve()
    receipt = verify_inputs(base)
    receiver, common, evidence = load_receiver_modules(base)
    spec = read_json(base / V1 / "registration.json")
    settings = evidence.Settings.from_registration(spec)
    if (settings.horizon, settings.template_duration, settings.event_false_alarm, spec["danger_squared"]) != (
            32, 10, 1 / BF_THRESHOLD, DANGER_SQUARED):
        raise ValueError("Unexpected registered detector, danger threshold or window.")
    frozen = read_json(base / FRONTIER / "results/comparison.json")["models"]
    report = {
        "baseline": BASELINE,
        "settings": {"horizon": settings.horizon, "attack_duration": settings.template_duration,
                     "bf_threshold": BF_THRESHOLD, "danger_squared": DANGER_SQUARED},
        "verified_frozen_files": len(receipt["files"]),
        "models": {},
    }
    details = {"timestamp_indexing": "0-based; null is right-censored at 32", "models": {}}
    for name in MODELS:
        config = frozen[name]["method_table"]
        names = [item["name"] for item in config]
        columns = [names.index(method) for method in METHODS]
        table = tuple(receiver.Method(**config[j]) for j in columns)
        replay, detail = replay_model(base, name, settings, spec, table, receiver, common, evidence)
        archive = stored_comparison(base, name, columns, settings.horizon, spec["evaluation_onset"])
        report["models"][name] = {"receiver_replay": replay, "saved_holdout_outcomes": archive}
        details["models"][name] = detail
    verify_inputs(base)
    return report, details


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent,
                        help="Root of the full frozen research tree")
    args = parser.parse_args()
    result, details = evaluate(args.baseline)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    write_json(output / "summary.json", result)
    write_json(output / "a0_replay_paths.json", details)
    for name, model in result["models"].items():
        rows = model["receiver_replay"]["cases"]["A0"]["methods"]
        counts = ", ".join(f"{m}: {r['event_counts']['before_alarm']}/{r['n']}"
                           for m, r in rows.items())
        print(f"{name} A0 strict pre-alarm danger from receiver replay: {counts}")


if __name__ == "__main__":
    main()
