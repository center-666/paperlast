"""Compare receiver policies on archived synthetic observations.

This program verifies and reads archive data. It does not execute the archive's
simulation, injection, or optimization scripts.
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from accommodation import CommonGPath, OnlineEvidence, PreparedModel, Settings, choose_soft_weight


HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE.parents[2] / "交付" / "paperlast_audit_20261113"
REFERENCE = Path("common_g_reference_revision_20261115")
A1 = Path("A1_information_aware_attack_audit_20261116")
SWITCH = Path("attack_gain_switch_audit_20261115")
METHODS = ("V0", "GLRT_reject", "shrink_G", "oracle_GLS", "scan_GLS", "V1a", "V1b")
METHODS_WITH_CORRECTION = ("oracle_GLS", "scan_GLS", "V1a", "V1b")
WEIGHT_DIAGNOSTICS = ("posterior_mean_weight", "uncertainty_limited_weight", "nominal_weight_limit")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def open_baseline(baseline, record):
    for name, expected in record["files"].items():
        path = baseline / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen baseline input differs: {name}")
    source = baseline / REFERENCE / "common_g_reference.py"
    spec = importlib.util.spec_from_file_location("verified_common_g_reference", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    return reference


@dataclass(frozen=True)
class Episode:
    states: np.ndarray
    local: np.ndarray
    healthy: np.ndarray
    clean_target: np.ndarray
    attacked_target: np.ndarray
    decoders: np.ndarray
    decoded_difference: np.ndarray
    energy: float


def archived_cases(baseline, reference, prepared, record):
    """Use the same original physical samples and decoder realizations in all cases."""
    settings, model = prepared.settings, prepared.model
    name = model.name
    indices = record["samples"]["physical_indices"]
    a1_registration = load_json(baseline / A1 / "registration.json")
    a1_rows = [a1_registration["validation_indices"].index(i) for i in indices]
    previous_selection = load_json(baseline / A1 / "selection_before_validation.json")["selection"][name]
    case_keys = {
        "A0_P3_proxy": "A0/P3_proxy_joint_GLRT",
        "A1_H_mismatch": "A1-H/P1_peak_mean_bias",
        "A1_C_mismatch": "A1-C/P1_peak_mean_bias",
    }
    with np.load(baseline / SWITCH / "screen_arrays.npz", allow_pickle=False) as bank:
        states = bank[f"{name}__physical__x"][indices]
        obs = bank[f"{name}__physical__y"][indices]
        decoders = bank[f"{name}__M_target"][indices]
        ordinary = bank[f"{name}__original_Gaussian__ordinary__attack"]
        archived_post = bank[f"{name}__original_Gaussian__ordinary__B0__post"][indices]
    cases = {
        "healthy": np.zeros(obs[:, :, settings.target].shape),
        "A0": np.broadcast_to(ordinary, obs[:, :, settings.target].shape)
    }
    with np.load(baseline / A1 / "validation_arrays.npz", allow_pickle=False) as bank:
        for case, selector in case_keys.items():
            key = previous_selection[selector].replace("/", "_")
            cases[case] = bank[f"{name}__{key}__a"][a1_rows]
            archived_bias = bank[f"{name}__{key}__b"][a1_rows]
            for index, (decoder, effect) in enumerate(zip(decoders, cases[case])):
                _, offset = reference.bias_recursion(model, settings.receiver, (decoder @ effect[..., None])[..., 0])
                np.testing.assert_allclose(offset, archived_bias[index], rtol=2e-10, atol=2e-10)
    result = {}
    for case, encoded_difference in cases.items():
        energy = np.sum(encoded_difference**2, axis=(1, 2))
        if np.max(energy) > settings.coded_energy_reference + 1e-9:
            raise ValueError(f"Archived record violates the declared input budget: {name} {case}")
        shift = (decoders @ encoded_difference[..., None])[..., 0]
        start = record["evaluation_onset"]
        stop = start + settings.template_duration
        if np.max(np.abs(shift[:, :start])) > 1e-12 or np.max(np.abs(shift[:, stop:])) > 1e-12:
            raise ValueError(f"Archived record has unexpected support: {name} {case}")
        original = obs[:, :, settings.target]
        result[case] = [
            Episode(
                states[i], obs[i, :, settings.receiver], obs[i, :, settings.healthy],
                original[i], original[i] + shift[i], decoders[i], shift[i], float(energy[i])
            ) for i in range(len(indices))
        ]
    return result, archived_post


def gain_on_nominal_grid(prepared, record):
    reference = prepared.nominal_mse().mean()
    limit = (1 + prepared.settings.nominal_one_step_allowance) * reference
    for scale in np.linspace(0.0, 1.0, record["gain_shrink_grid_intervals"] + 1):
        if prepared.nominal_mse(neighbor_scale=scale).mean() <= limit + 1e-14:
            return float(scale)
    raise AssertionError("The unchanged gain must be feasible.")


def run_episode(prepared, common, episode, record, scale, parity):
    """Advance each policy's own posterior on the same received packet stream."""
    p, s = prepared, prepared.settings
    path = {name: CommonGPath(p, common, scale if name == "shrink_G" else 1.0)
            for name in METHODS}
    observer = OnlineEvidence(p, common)
    posterior = {name: np.zeros((s.horizon, p.n)) for name in METHODS}
    alarms = {name: np.zeros(s.horizon, dtype=bool) for name in METHODS}
    estimates = {name: np.zeros((s.horizon, p.m)) for name in METHODS_WITH_CORRECTION}
    weights = {name: np.zeros(s.horizon) for name in METHODS_WITH_CORRECTION}
    correction = {name: np.zeros((s.horizon, p.m)) for name in METHODS_WITH_CORRECTION}
    trace = {name: np.zeros(s.horizon) for name in ("nominal_risk", "nominal_increase", "robust_increase")}
    trace.update({name: np.zeros(s.horizon) for name in WEIGHT_DIAGNOSTICS})
    trace["target_rejected"] = np.zeros(s.horizon, dtype=bool)
    trace["probabilities"] = np.zeros((s.horizon, s.horizon + 1))
    trace["delta_variance"] = np.zeros(s.horizon)
    trace["module_runtime_ns"] = np.zeros((s.horizon, 3), dtype=np.int64)
    trace["reference_posterior"] = np.zeros((s.horizon, p.n))
    trace["oracle_Bayes_delta"] = np.zeros((s.horizon, p.m))
    target_y = episode.attacked_target
    order, z, chol_theta, threshold = parity
    hold = 0
    oracle_start = record["evaluation_onset"]
    for k in range(s.horizon):
        local, healthy, target = episode.local[k], episode.healthy[k], target_y[k]
        time = perf_counter_ns()
        evidence = observer.observe(local, healthy, target, episode.decoders[k])
        trace["module_runtime_ns"][k, 0] = perf_counter_ns() - time
        trace["reference_posterior"][k] = evidence.reference_posterior
        trace["probabilities"][k] = evidence.weights
        trace["delta_variance"][k] = np.trace(evidence.delta_cov)

        parity_alarm = False
        if k >= order:
            zoned = z @ target_y[k - order:k + 1].reshape(-1)
            white = solve_triangular(chol_theta, zoned, lower=True)
            parity_alarm = bool(white @ white > threshold)
        alarms["V0"][k] = alarms["shrink_G"][k] = parity_alarm
        alarms["GLRT_reject"][k] = alarms["scan_GLS"][k] = alarms["oracle_GLS"][k] = evidence.matched_alarm
        alarms["V1a"][k] = alarms["V1b"][k] = evidence.mixture_alarm

        if evidence.matched_alarm:
            hold = record["reject_hold_updates"]
        blocked = hold > 0
        trace["target_rejected"][k] = blocked
        hold = max(0, hold - 1)
        time = perf_counter_ns()
        posterior["V0"][k] = path["V0"].update(local, healthy, target)
        trace["module_runtime_ns"][k, 1] = perf_counter_ns() - time
        posterior["shrink_G"][k] = path["shrink_G"].update(local, healthy, target)
        posterior["GLRT_reject"][k] = path["GLRT_reject"].update(
            local, healthy, target, accept_target=not blocked
        )

        if oracle_start <= k < oracle_start + s.template_duration:
            estimates["oracle_GLS"][k] = evidence.fits[oracle_start].gls_delta
            weights["oracle_GLS"][k] = 1.0
            trace["oracle_Bayes_delta"][k] = evidence.fits[oracle_start].delta
        if evidence.matched_alarm:
            estimates["scan_GLS"][k] = evidence.fits[evidence.matched_start].gls_delta
            weights["scan_GLS"][k] = 1.0
        estimates["V1a"][k] = estimates["V1b"][k] = evidence.delta
        weights["V1a"][k] = 1.0
        for name in ("oracle_GLS", "scan_GLS", "V1a"):
            correction[name][k] = weights[name][k] * estimates[name][k]
            posterior[name][k] = path[name].update(local, healthy, target - correction[name][k])

        time = perf_counter_ns()
        preview = path["V1b"].preview(local, healthy, target)
        decision = choose_soft_weight(preview, evidence, p.model.G[s.receiver], s)
        weights["V1b"][k] = decision.alpha
        correction["V1b"][k] = decision.alpha * evidence.delta
        posterior["V1b"][k] = path["V1b"].update(local, healthy, target - correction["V1b"][k])
        trace["module_runtime_ns"][k, 2] = perf_counter_ns() - time
        trace["nominal_risk"][k] = decision.risk_without_correction
        trace["nominal_increase"][k] = decision.nominal_risk_increase
        trace["robust_increase"][k] = decision.robust_model_risk_increase
        for field in WEIGHT_DIAGNOSTICS:
            trace[field][k] = getattr(decision, field)
        for name, path_states in posterior.items():
            if not np.all(np.isfinite(path_states[k])):
                raise FloatingPointError(f"Nonfinite output in {name} at {k}.")

    return dict(posterior=posterior, alarms=alarms, estimated=estimates,
                alpha=weights, correction=correction, trace=trace)


def first(mask):
    """One event per episode. The array length denotes no event."""
    index = np.flatnonzero(mask)
    return int(index[0]) if index.size else len(mask)


def summarize(paths, clean_paths, episodes, record, name):
    start = record["evaluation_onset"]
    horizon = record["horizon"]
    stop = start + record["template_duration"]
    state = np.stack([p.states for p in episodes])
    nominal = np.stack([p["posterior"][name] for p in clean_paths])
    estimate = np.stack([p["posterior"][name] for p in paths])
    alarms = np.stack([p["alarms"][name] for p in paths])
    b = estimate - nominal
    e0 = nominal - state
    e = estimate - state
    nominal_sq = np.sum(e0**2, axis=2)
    attack_sq = np.sum(e**2, axis=2)
    bias_sq = np.sum(b**2, axis=2)
    cross = 2 * np.sum(e0 * b, axis=2)
    np.testing.assert_allclose(attack_sq, nominal_sq + bias_sq + cross, rtol=1e-12, atol=1e-12)
    a = np.array([first(row) for row in alarms])
    d = np.array([first(row >= record["danger_squared"]) for row in bias_sq])
    risk = d < a
    predetect = np.arange(horizon)[None, :] < a[:, None]
    within = (a >= start) & (a < stop)
    onsets = [int(x) if x < horizon else None for x in a]
    losses = attack_sq[:, start:stop].mean(axis=1)
    active_before = predetect[:, start:stop]
    result = {
        "paths": len(episodes),
        "healthy_mse": float(nominal_sq.mean()),
        "healthy_false_alarms": int(sum(np.any(p["alarms"][name]) for p in clean_paths)),
        "episode_mse": float(losses.mean()),
        "episode_bias_power": float(bias_sq[:, start:stop].mean()),
        "episode_cross_term": float(cross[:, start:stop].mean()),
        "episode_pre_alarm_mse": float(attack_sq[:, start:stop][active_before].mean()) if active_before.any() else None,
        "episode_post_alarm_mse": float(attack_sq[:, start:stop][~active_before].mean()) if (~active_before).any() else None,
        "episode_pre_alarm_updates": int(active_before.sum()),
        "cumulative_paired_bias": float(bias_sq[:, start:].sum(axis=1).mean()),
        "first_packet_danger": int(np.sum(bias_sq[:, start] >= record["danger_squared"])),
        "danger_before_alarm": int(risk.sum()),
        "first_alarms_in_attacked_support": int(within.sum()),
        "alarms_before_support": int((a < start).sum()),
        "first_alarm_timestamps": onsets,
        "median_first_alarm_delay_given_in_support": float(np.median(a[within] - start)) if within.any() else None,
        "tail_paired_bias": float(bias_sq[:, -5:].mean()),
        "last_paired_bias": float(bias_sq[:, -1].mean()),
        "paired_path_mse": losses.tolist(),
        "estimate_mse": None,
        "uncompensated_measurement_error_mse": None,
        "mean_applied_weight": None
    }
    if name in METHODS_WITH_CORRECTION:
        true_shift = np.stack([p.decoded_difference for p in episodes])
        pred_shift = np.stack([p["estimated"][name] for p in paths])
        corrected = np.stack([p["correction"][name] for p in paths])
        w = np.stack([p["alpha"][name] for p in paths])
        result["estimate_mse"] = float(np.sum((pred_shift[:, start:stop] - true_shift[:, start:stop])**2, axis=2).mean())
        result["uncompensated_measurement_error_mse"] = float(np.sum(
            (corrected[:, start:stop] - true_shift[:, start:stop])**2, axis=2
        ).mean())
        result["mean_applied_weight"] = float(w[:, start:stop].mean())
        result["positive_weight_fraction"] = float(np.mean(w[:, start:stop] > 0))
    return result


def structural_diagnostics(prepared, paths, episodes, record):
    """Retrospective attribution. The actual receiver never uses these labels."""
    start = record["evaluation_onset"]
    stop = start + prepared.settings.template_duration
    m, n = prepared.m, prepared.n
    chol = prepared.prefixes[stop - 1].target_cholesky
    mismatch = []
    for episode in episodes:
        design = np.zeros((stop, m, n))
        design[start:stop] = episode.decoders[start:stop] @ prepared.template
        white = solve_triangular(chol, design.reshape(-1, n), lower=True)
        basis, _ = np.linalg.qr(white, mode="reduced")
        recorded_shift = solve_triangular(chol, episode.decoded_difference[:stop].reshape(-1), lower=True)
        contrast = recorded_shift - basis @ (basis.T @ recorded_shift)
        shift_energy = recorded_shift @ recorded_shift
        mismatch.append(float((contrast @ contrast) / shift_energy) if shift_energy > 0 else 0.0)
    oracle = np.stack([p["trace"]["oracle_Bayes_delta"][start:stop] for p in paths])
    measured = np.stack([p.decoded_difference[start:stop] for p in episodes])
    factors = {field: np.stack([p["trace"][field][start:stop] for p in paths])
               for field in WEIGHT_DIAGNOSTICS}
    mean = factors["posterior_mean_weight"]
    conservative = factors["uncertainty_limited_weight"]
    limit = factors["nominal_weight_limit"]
    return {
        "mean_whitened_off_template_energy_fraction_at_end": float(np.mean(mismatch)),
        "oracle_Bayes_delta_mse_audit_only": float(np.sum((oracle - measured)**2, axis=2).mean()),
        "weights_on_actual_V1b_histories": {
            "average_posterior_mean_weight": float(mean.mean()),
            "average_uncertainty_limited_weight": float(conservative.mean()),
            "average_final_weight": float(np.minimum(conservative, limit).mean()),
            "fraction_nominal_bound_strictly_active": float(np.mean(limit < conservative - 1e-12)),
        }
    }


def run_model(baseline, common, settings, record, name, output_arrays):
    model = common.load_model(name, baseline / REFERENCE / "inputs" / "five_node_aggregate_model.json")
    begin = perf_counter_ns()
    prepared = PreparedModel(model, settings)
    setup_ms = (perf_counter_ns() - begin) / 1e6
    cases, ordinary_baseline = archived_cases(baseline, common, prepared, record)
    scale = gain_on_nominal_grid(prepared, record)
    order = record["parity_order"]
    _, z, _, theta = common.parity_model(model.A, model.H[settings.target],
                                         model.Q, model.R[settings.target], order)
    threshold = chi2.isf(settings.event_false_alarm / (settings.horizon - order), len(theta))
    parity = order, z, np.linalg.cholesky(theta), threshold
    clean = [run_episode(prepared, common.common_step, p, record, scale, parity) for p in cases["healthy"]]
    output = {}
    runtime = []
    start = record["evaluation_onset"]
    stop = start + settings.template_duration
    max_guard = 0.0
    max_risk = 0.0
    for case, episodes in cases.items():
        paths = clean if case == "healthy" else [
            run_episode(prepared, common.common_step, p, record, scale, parity) for p in episodes
        ]
        if case == "A0":
            np.testing.assert_allclose(np.stack([p["posterior"]["V0"] for p in paths]), ordinary_baseline,
                                       rtol=2e-10, atol=2e-10)
        for p in paths:
            t = p["trace"]
            max_guard = max(max_guard, float(np.max(t["nominal_increase"] - settings.nominal_one_step_allowance * t["nominal_risk"])))
            max_risk = max(max_risk, float(np.max(t["robust_increase"])))
            runtime.append(t["module_runtime_ns"])
        assert max_guard < 1e-10 and max_risk < 1e-10
        methods = {m: summarize(paths, clean, episodes, record, m) for m in METHODS}
        for m in METHODS:
            summary = methods[m]
            diff = np.array(summary["paired_path_mse"]) - np.array(methods["V0"]["paired_path_mse"])
            summary["mse_difference_vs_V0"] = float(diff.mean())
            summary["paired_standard_error"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            for field in ("posterior", "alarms"):
                output_arrays[f"{name}__{case}__{m}__{field}"] = np.stack([p[field][m] for p in paths])
            if m in METHODS_WITH_CORRECTION:
                for field in ("alpha", "estimated", "correction"):
                    output_arrays[f"{name}__{case}__{m}__{field}"] = np.stack([p[field][m] for p in paths])
        for field in ("nominal_risk", "nominal_increase", "robust_increase", "probabilities",
                      "target_rejected", "delta_variance", "reference_posterior",
                      "oracle_Bayes_delta", *WEIGHT_DIAGNOSTICS):
            output_arrays[f"{name}__{case}__{field}"] = np.stack([p["trace"][field] for p in paths])
        output_arrays[f"{name}__states"] = np.stack([p.states for p in episodes])
        weights_at_end = np.array([p["trace"]["probabilities"][stop - 1, start + 1] for p in paths])
        output[case] = {
            "max_recorded_coded_energy": max(p.energy for p in episodes),
            "mean_posterior_mass_on_recorded_onset_at_end" : float(weights_at_end.mean()) if case != "healthy" else None,
            "retrospective_diagnostics": structural_diagnostics(prepared, paths, episodes, record),
            "methods": methods,
        }
        line = " ".join(f"{m}={methods[m]['episode_mse']:.4f}" for m in METHODS)
        print(f"{name} {case}: {line}", flush=True)

    times = np.concatenate(runtime, axis=0) / 1000
    trusted = prepared.nominal_mse(target_accepted=False)
    baseline_normal = prepared.nominal_mse()
    return {
        "F2": float(np.linalg.norm(model.F(settings.receiver), 2)),
        "reference_first_prior_target_cross": prepared.reference_target_cross[0, 0].tolist(),
        "max_reference_target_cross_norm": float(max(
            np.linalg.norm(prepared.reference_target_cross[k, k]) for k in range(settings.horizon)
        )),
        "analytic_health_mse_V0": float(baseline_normal.mean()),
        "analytic_health_mse_trusted_only": float(trusted.mean()),
        "analytic_health_mse_shrink_G": float(prepared.nominal_mse(neighbor_scale=scale).mean()),
        "shrink_G_scale": scale,
        "matched_tests": prepared.active_test_count,
        "nominal_parity_threshold": float(threshold),
        "max_one_step_guard_violation": max_guard,
        "max_model_robust_risk_increase": max_risk,
        "setup_ms": setup_ms,
        "timing_median_us": dict(zip(("joint_inference_including_GLS", "V0_update_only", "soft_update_only"), np.median(times, axis=0))),
        "timing_p95_us": dict(zip(("joint_inference_including_GLS", "V0_update_only", "soft_update_only"), np.quantile(times, 0.95, axis=0))),
        "cases": output
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    record = load_json(HERE / "registration.json")
    common = open_baseline(args.baseline, record)
    settings = Settings.from_registration(record)
    arrays = {}
    models = {name: run_model(args.baseline, common, settings, record, name, arrays)
              for name in record["models"]}
    result = {
        "baseline": record["baseline"],
        "registration_sha256": hashlib.sha256((HERE / "registration.json").read_bytes()).hexdigest(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "frozen_input_hashes_verified": len(record["files"]),
        "physical_samples_added": 0,
        "whole_policy_attack_optimality": False,
        "models": models
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n"
    )
    np.savez_compressed(args.out / "paired_paths.npz", **arrays)


if __name__ == "__main__":
    main()
