"""Read-only A0 model audit, prefix theory evaluation, and nominal-cost replay."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

from early_experiment import posterior_checks, prefix_report


HERE = Path(__file__).resolve().parent
BASELINE = "c050b2033589868145878ff2047e733d47e68582"
OFFLINE = "JFI_defensive_offline_validation"
V12_NOMINAL = Path("JFI_V12_budget_and_risk_aware_compensation_20261118/results/nominal_audit.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pinned_dependencies(base):
    lock = read_json(HERE / "dependency_lock.json")
    if lock["baseline"] != BASELINE:
        raise ValueError("Unexpected baseline pin.")
    for relative, expected in lock["files"].items():
        raw = (base / relative).read_bytes().replace(b"\r\n", b"\n")
        if sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen input differs: {relative}")
    return len(lock["files"])


def model_audit(p, spec, stored, onset_counts):
    s, length = spec["evaluation_onset"], p.settings.template_duration
    source = stored[s:s+length].copy()
    np.testing.assert_array_equal(stored[:s], 0.)
    np.testing.assert_array_equal(stored[s+length:], 0.)
    if np.linalg.norm(source[0]) <= 1e-12:
        raise ValueError("The two-start strictness proof uses a nonzero first profile term.")
    design = p.template.reshape(-1, p.n)
    fit = np.linalg.lstsq(design, source.reshape(-1), rcond=None)[0]
    delta = design @ fit - source.reshape(-1)
    np.testing.assert_allclose(delta, 0, atol=2e-12)
    budget = p.settings.coded_energy_reference
    actual_energy = float(np.sum(source**2))
    np.testing.assert_allclose(actual_energy, budget, atol=2e-11, rtol=0)
    q = p.parameter_prior_cholesky
    eigenvalues = np.linalg.eigvalsh((design @ q).T @ (design @ q))
    np.testing.assert_allclose(eigenvalues, budget/p.n, atol=2e-11)
    return source, {
        "test_parameter_law": "Point mass for each model. One archived full-energy ordinary profile.",
        "coded_energy_verified": actual_energy,
        "deterministic_template_max_residual": float(np.max(np.abs(delta))),
        "test_support_fixed_start": [s, s+length-1],
        "saved_holdout_start_counts": onset_counts,
        "holdout_registered_start_law_conditional_on_event": {"3":1/3, "7":1/3, "11":1/3},
        "working_prior_no_event": p.settings.no_event_prior,
        "working_event_start_support": [0, p.settings.horizon-1],
        "working_event_start_law": "Uniform conditional on event; future prefixes have unit BF.",
        "working_gaussian_coded_energy_mean": budget,
        "working_energy_quadratic_form_eigenvalues": eigenvalues.tolist(),
        "working_gaussian_probability_full_template_energy_exceeds_budget": float(chi2.sf(p.n, p.n)),
    }


def analyze_model(base, name, old, receiver, common, evidence, spec, earlier, frozen, bridge, capture):
    settings = evidence.Settings.from_registration(spec)
    p = evidence.PreparedModel(common.load_model(name, base / old.MODEL), settings)
    table = {row["name"]: row for row in frozen["method_table"]}
    frontier = import_module("frontier")
    match = bridge.covariance_match(p, frontier, frozen["cases"]["healthy"]["methods"], earlier, table)
    methods = tuple(receiver.Method(**table[key]) for key in ("V0", "V1a", *bridge.POINTS))
    methods += (receiver.Method(bridge.MATCHED, "anchor", match["matched_radius_for_point_estimate"]),)
    ids = spec["samples"]["physical_indices"]
    with np.load(base / old.OBSERVATIONS, allow_pickle=False) as saved:
        stored = saved[f"{name}__original_Gaussian__ordinary__attack"]
        decoder = saved[f"{name}__M_target"]
        states = saved[f"{name}__physical__x"][ids]
        v0_offsets = (saved[f"{name}__original_Gaussian__ordinary__B0__post"]
                      - saved[f"{name}__normal__B0__post"])
    with np.load(base / old.FRONTIER / f"results/{name}_paired.npz", allow_pickle=False) as saved:
        vals, counts = np.unique(saved["unknown_start__onset"], return_counts=True)
    onset_counts = {str(val): int(count) for val, count in zip(vals, counts, strict=True)}
    profile, audit = model_audit(p, spec, stored, onset_counts)
    f, gain = p.model.F(settings.receiver), p.model.G[settings.receiver]
    decoded_archive = np.einsum("ntij,tj->nti", decoder, stored)
    previous_offsets = np.concatenate((np.zeros_like(v0_offsets[:, :1]), v0_offsets[:, :-1]), axis=1)
    np.testing.assert_allclose(v0_offsets, previous_offsets @ f.T + decoded_archive @ gain.T,
                               atol=2e-12, rtol=2e-12)
    prefix = prefix_report(p, spec["watermark"], profile, decoder, v0_offsets)
    trusted, target, key = old.saved_sensor_inputs(base, name, settings, ids)
    with np.load(base / old.V11 / f"phase1_results/{name}_step_logs.npz", allow_pickle=False) as logs:
        decoded = logs["A0__decoded_difference"]
        np.testing.assert_array_equal(states, logs["A0__state_truth"])
        np.testing.assert_array_equal(states, logs["healthy__state_truth"])
        np.testing.assert_allclose(
            np.einsum("ntij,tj->nti", key, stored), decoded, atol=2e-12, rtol=2e-12
        )
        clean_obs = receiver.Observations(trusted, target, key)
        disturbed = (target + decoded)
        disturbed.flags.writeable = False
        obs = receiver.Observations(trusted, disturbed, key)
        clean = receiver.run_receiver(p, clean_obs, methods)
        active = receiver.run_receiver(p, obs, methods)
        observer = receiver.BatchedObserver(p)
        trace, control_trace = capture(observer, obs), capture(observer, clean_obs)
        names = old.read_json(base / old.V11 / "phase1_results/ablation.json")["policies"]
        regression = {
            "A0": old.compare_replay_with_log(active, trace, logs, "A0", names),
            "healthy": old.compare_replay_with_log(clean, control_trace, logs, "healthy", names),
        }
        work_covariance = logs["A0__delta_cov"]

    onset = np.full(len(ids), spec["evaluation_onset"])
    stop = onset + settings.template_duration
    policies = tuple(method.name for method in methods)
    replay, events = bridge.matched_replay(
        "Exploratory scoring on the 12 previously stored complete A0 records per model",
        clean.post, active.post, states, onset, stop, active.log_bf, policies
    )
    bridge_result = bridge.propagation_report(p, active.post[:, 1], clean.post[:, 1], states, decoded,
                                              trace, control_trace, work_covariance, events)
    return {
        "model_consistency": audit,
        "first_three_prefixes": prefix,
        "posterior_audit": posterior_checks(p, trace, obs, profile),
        "common_g_bridge": bridge_result,
        "nominal_matching": match,
        "matched_replay": replay,
        "frozen_replay_max_absolute_error": max(max(d.values()) for d in regression.values()),
    }


def evaluate(base=HERE.parent):
    base = Path(base).resolve()
    checked = pinned_dependencies(base)
    roots = sys.path[:]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(base/OFFLINE))
    try:
        old = import_module("replay_evaluator")
        if Path(old.__file__).resolve() != base/OFFLINE/"replay_evaluator.py":
            raise ValueError("Unexpected frozen evaluator import.")
        lock = old.verify_inputs(base)
        receiver, common, evidence = old.load_receiver_modules(base)
        capture = import_module("onset_diagnostics").capture_posterior
        bridge = import_module("bridge_and_cost")
        spec = old.read_json(base / old.V1 / "registration.json")
        conditions = tuple(spec[key] for key in ("evaluation_onset", "template_duration", "horizon",
                                                "no_event_prior", "coded_energy_reference", "event_false_alarm",
                                                "danger_squared"))
        if conditions != (7, 10, 32, .5, 10, .05, .1):
            raise ValueError("A0 study assumes the frozen protocol.")
        source = read_json(base / old.FRONTIER / "results/comparison.json")["models"]
        previous = read_json(base / V12_NOMINAL)
        models = {
            name: analyze_model(base, name, old, receiver, common, evidence, spec, previous[name],
                                source[name], bridge, capture)
            for name in old.MODELS
        }
        old.verify_inputs(base)
        pinned_dependencies(base)
    finally:
        sys.path[:] = roots
    return {
        "baseline": BASELINE,
        "frozen_original_inputs": len(lock["files"]),
        "additional_pinned_inputs": checked,
        "new_physical_paths": 0,
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=HERE.parent, type=Path, help="Root of the frozen research tree")
    args = parser.parse_args()
    results = evaluate(args.baseline)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, allow_nan=False, indent=2)+"\n",
                                      encoding="utf-8", newline="\n")
    for name, model in results["models"].items():
        metrics = [row["fixed_start_7_equal_prior"]["bayes_error_saved_decoder_average"]
                   for row in model["first_three_prefixes"].values()]
        radius = model["nominal_matching"]["matched_radius_for_point_estimate"]
        print(f"{name}: first-three equal-prior reference errors {np.round(metrics, 6).tolist()}; "
              f"nominal matching radius {radius:.8f} on saved inputs")


if __name__ == "__main__":
    main()
