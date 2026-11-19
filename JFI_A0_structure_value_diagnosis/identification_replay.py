"""Post-run A0 posterior diagnostics on existing, complete receiver inputs."""

from itertools import combinations

import numpy as np

from archive_analysis import POLICIES
from onset_diagnostics import capture_posterior
from paired_analysis import defender_comparison, paired_risk
from replay_evaluator import (
    OBSERVATIONS, V11, MODEL, compare_replay_with_log, read_json, saved_sensor_inputs,
)
from risk_metrics import BF_THRESHOLD, DANGER_SQUARED, masked_mean, score_estimates


ACTIVE_LABEL = "activity-label diagnostic"
ONSET_LABEL = "true-onset diagnostic"
SATURATION_REFERENCE = "anchor@2%"


def label_references(trace, onset, stop):
    """Extract already fitted components for offline label-aware diagnostics."""
    n, horizon, _ = trace["delta"].shape
    ticks = np.arange(horizon)[None, :]
    active = (ticks >= onset[:, None]) & (ticks < stop[:, None])
    mass = trace["p_active"]
    if np.any(mass[active] <= 0):
        raise ValueError("Active conditioning needs positive posterior mass on the stored support.")
    activity_mean = np.divide(trace["delta"], mass[..., None],
                              out=np.zeros_like(trace["delta"]), where=active[..., None])
    true_mean = trace["component_mean"][np.arange(n)[:, None], ticks, onset[:, None]]
    return activity_mean, np.where(active[..., None], true_mean, 0.0)


def decomposition(trace, activity_mean, true_mean, true_distortion, onset, stop, gain):
    """Retain the three square terms and each signed pairwise cross term."""
    ticks = np.arange(trace["delta"].shape[1])[None, :]
    offset = ticks - onset[:, None]
    active = (offset >= 0) & (ticks < stop[:, None])
    phases = {"first_packet": offset == 0, "first_three": active & (offset < 3),
              "remaining_seven": active & (offset >= 3), "whole_support": active}
    components = {
        "inactive_mass": trace["delta"] - activity_mean,
        "onset_mixture": activity_mean - true_mean,
        "true_onset_fit": true_mean - true_distortion,
    }
    np.testing.assert_allclose(sum(components.values()), trace["delta"] - true_distortion,
                               atol=2e-12, rtol=2e-12)

    def squared_terms(vectors, selected):
        values = {name: vector[selected] for name, vector in vectors.items()}
        squares = {name: float(np.mean(np.sum(vector**2, axis=-1)))
                   for name, vector in values.items()}
        cross = {f"{a} x {b}": float(2 * np.mean(np.sum(x*y, axis=-1)))
                 for (a, x), (b, y) in combinations(values.items(), 2)}
        total = float(np.mean(np.sum(sum(values.values())**2, axis=-1)))
        np.testing.assert_allclose(total, sum(squares.values()) + sum(cross.values()), atol=1e-12)
        return {"squares": squares, "signed_cross": cross, "total": total}

    out = {}
    true_weight = trace["weights"][np.arange(len(onset))[:, None], ticks, onset[:, None] + 1]
    state_terms = {name: value @ gain.T for name, value in components.items()}
    for label, selection in phases.items():
        out[label] = {
            "packets": int(selection.sum()),
            "no_event_probability": float(trace["weights"][:, :, 0][selection].mean()),
            "active_probability": float(trace["p_active"][selection].mean()),
            "true_onset_probability": float(true_weight[selection].mean()),
            "decoded_distortion": squared_terms(components, selection),
            "current_update_units": squared_terms(state_terms, selection),
        }
    return out


def replay_label_reference(prepared, observations, compensations, path_type, step):
    """Use the unmodified common-G update on a finite saved record."""
    n, horizon = len(observations), prepared.settings.horizon
    m = prepared.m
    output = np.empty((n, horizon, prepared.n))
    for i in range(n):
        path = path_type(prepared, step)
        for k in range(horizon):
            trusted = observations.trusted[i, k]
            output[i, k] = path.update(trusted[:m], trusted[m:],
                                       observations.target[i, k] - compensations[i, k])
    return output


def fixed_history_summary(attacked, clean, truth, onset, stop, original, path_ids):
    """Score individual replacements of the update on the same V1a histories."""
    ticks = np.arange(attacked.shape[1])[None, :]
    active = (ticks >= onset[:, None]) & (ticks < stop[:, None])
    first = ticks == onset[:, None]
    dangerous = np.sum((attacked - clean)**2, axis=2) >= DANGER_SQUARED
    strict = dangerous & (ticks < original.events.alarm[:, None])
    old_index = np.minimum(original.events.danger, attacked.shape[1] - 1)
    still = strict[np.arange(len(onset)), old_index]
    rescued = original.events.before & ~still
    squared_error = np.sum((attacked - truth)**2, axis=2)
    return {
        "support_step_mse": float(masked_mean(squared_error, active).mean()),
        "first_packet_mse": float(squared_error[first].mean()),
        "first_packet_danger": int(np.sum(dangerous & first)),
        "first_packet_strict_pre_alarm": int(np.sum(strict & first)),
        "strict_pre_alarm_dangerous_support_ticks": int(np.sum(strict & active)),
        "original_strict_danger_tick_now_below_threshold_paths": np.asarray(path_ids)[rescued].tolist(),
    }


def inspect_complete_records(base, model, method_table, receiver, common, evidence, spec):
    settings = evidence.Settings.from_registration(spec)
    prepared = evidence.PreparedModel(common.load_model(model, base / MODEL), settings)
    gain = prepared.model.G[settings.receiver]
    table = tuple(receiver.Method(**method_table[name]) for name in POLICIES)
    path_ids = spec["samples"]["physical_indices"]
    trusted, target, decoder = saved_sensor_inputs(base, model, settings, path_ids)
    legacy = read_json(base / V11 / "phase1_results/ablation.json")["policies"]
    with np.load(base / V11 / f"phase1_results/{model}_step_logs.npz", allow_pickle=False) as logs:
        shift = logs["A0__decoded_difference"]
        recorded_target = target + shift
        recorded_target.flags.writeable = False
        obs0 = receiver.Observations(trusted, target, decoder)
        obs = receiver.Observations(trusted, recorded_target, decoder)

        nominal = receiver.run_receiver(prepared, obs0, table)
        run = receiver.run_receiver(prepared, obs, table)
        observer = receiver.BatchedObserver(prepared)
        trace = capture_posterior(observer, obs)
        trace0 = capture_posterior(observer, obs0)
        checks = {
            "A0": compare_replay_with_log(run, trace, logs, "A0", legacy),
            "healthy": compare_replay_with_log(nominal, trace0, logs, "healthy", legacy),
        }
        with np.load(base / OBSERVATIONS, allow_pickle=False) as raw:
            truth = raw[f"{model}__physical__x"][path_ids]
        np.testing.assert_array_equal(truth, logs["A0__state_truth"])
        np.testing.assert_array_equal(truth, logs["healthy__state_truth"])

    # Labels enter only after the receiver and posterior runs finish.
    onset = np.full(len(path_ids), spec["evaluation_onset"])
    stop = onset + settings.template_duration
    active_mean, start_mean = label_references(trace, onset, stop)
    analysis = decomposition(trace, active_mean, start_mean, shift, onset, stop, gain)
    fixed_posterior = {}
    full_metrics = {
        policy: score_estimates(run.post[:, j], nominal.post[:, j], truth, run.log_bf, onset, stop)
        for j, policy in enumerate(POLICIES)
    }
    v1a = full_metrics["V1a"]
    v = POLICIES.index("V1a")
    sat = POLICIES.index(SATURATION_REFERENCE)
    preview_a = run.post[:, v] + run.change[:, v]
    preview_0 = nominal.post[:, v] + nominal.change[:, v]
    fixed_posterior["V1a"] = (run.post[:, v], nominal.post[:, v])
    # The frozen saturation output projects the observation-only conditional mean.
    fixed_posterior[SATURATION_REFERENCE] = (run.post[:, sat], nominal.post[:, sat])

    label_clean = replay_label_reference(prepared, obs0, np.zeros_like(shift),
                                         evidence.CommonGPath, common.common_step)
    np.testing.assert_allclose(label_clean, nominal.post[:, POLICIES.index("V0")],
                               atol=2e-12, rtol=2e-12)
    extra = {}
    for label, compensations in ((ACTIVE_LABEL, active_mean), (ONSET_LABEL, start_mean)):
        fixed_posterior[label] = (preview_a - compensations @ gain.T, preview_0)
        post = replay_label_reference(prepared, obs, compensations, evidence.CommonGPath, common.common_step)
        full_metrics[label] = score_estimates(post, label_clean, truth, run.log_bf, onset, stop)
        full_first = full_metrics[label].events.danger == onset
        fixed_first = np.sum((fixed_posterior[label][0] - preview_0)**2, axis=-1) >= DANGER_SQUARED
        extra[label] = {
            "max_difference_fixed_vs_propagated_output": float(np.max(np.abs(post-fixed_posterior[label][0]))),
            "closed_loop_vs_v1a": paired_risk(full_metrics[label].events.before, v1a.events.before),
            "closed_loop_first_packet_count": int(full_first.sum()),
            "fixed_first_packet_count": int(fixed_first[np.arange(len(onset)), onset].sum()),
        }
    fixed_summary = {
        method: fixed_history_summary(a, h, truth, onset, stop, v1a, path_ids)
        for method, (a, h) in fixed_posterior.items()
    }

    details = {}
    for method in ("V1a", ACTIVE_LABEL, ONSET_LABEL, SATURATION_REFERENCE):
        row = full_metrics[method]
        details[method] = [
            {
                "path_id": int(pid),
                "support_mse": float(row.support_mse[i]),
                "tau_d": None if row.events.danger[i] == settings.horizon else int(row.events.danger[i]),
                "tau_a": None if row.events.alarm[i] == settings.horizon else int(row.events.alarm[i]),
                "strict_pre": bool(row.events.before[i]),
                "first_packet": bool(row.events.first_packet[i]),
            }
            for i, pid in enumerate(path_ids)
        ]
    return {
        "provenance": "full receiver rerun using the saved V1/V1.1 observations and decoded differences",
        "path_count": len(path_ids),
        "path_ids": path_ids,
        "regression_max_absolute_error": max(max(c.values()) for c in checks.values()),
        "label_oracle_healthy_equals_v0_max_abs": float(np.max(np.abs(label_clean - nominal.post[:, 0]))),
        "frozen_log_regression": checks,
        "decomposition": analysis,
        "fixed_history": fixed_summary,
        "closed_loop": defender_comparison(full_metrics),
        "label_diagnostics": extra,
        "closed_loop_paths": details,
        "common_bf_threshold": BF_THRESHOLD,
    }
