"""Paired common-G propagation and nominal-only saturation matching."""

import numpy as np
from scipy.stats import t

from paired_analysis import defender_comparison
from risk_metrics import DANGER_SQUARED, score_estimates


POINTS = ("anchor@0.1%", "anchor@0.5%", "anchor@1%", "anchor@2%")
MATCHED = "nominal-matched saturation"


def squared(rows):
    return np.sum(rows**2, axis=-1)


def covariance_match(prepared, frontier, source, previous_nominal, method_table):
    full, trusted, eigenvalues = frontier.covariances(prepared)
    floor = float(np.trace(full, axis1=1, axis2=2).mean())
    ceiling = float(np.trace(trusted, axis1=1, axis2=2).mean())
    full_trace = np.trace(full, axis1=1, axis2=2)
    v0 = source["V0"]["health_expected_MSE_exact_formula"]
    pct, low, high = source["V1a"]["health_expectation_pct_conditional_MC95"]
    center, lo, hi = v0*(1 + np.array([pct, low, high])/100)
    radii = [float(frontier.radius_for_cost(value, full_trace, eigenvalues))
             for value in (center, hi, lo)]
    previous_target = previous_nominal["policies"]["V1a"]["average_mse_model_estimate"]
    independent_radius = frontier.radius_for_cost(previous_target, full_trace, eigenvalues)
    matched_cost = float(frontier.rms_cost(radii[0], full_trace, eigenvalues).mean())
    np.testing.assert_allclose(center, matched_cost, atol=5e-13, rtol=0)
    other = {}
    for name in POINTS:
        radius = method_table[name]["parameter"]
        cost = float(frontier.rms_cost(radius, full_trace, eigenvalues).mean())
        np.testing.assert_allclose(cost, source[name]["health_expected_MSE_exact_formula"], atol=2e-12)
        other[name] = {"radius_frozen": radius, "nominal_health_mse": cost}
    return {
        "calibration_source": "Only frozen healthy Gaussian covariances and earlier healthy conditional MSE",
        "v0_healthy_mse": v0,
        "full_information_mse_at_infinite_radius": floor,
        "trusted_only_mse_at_zero_radius": ceiling,
        "v1a_nominal_target_estimate": float(center),
        "v1a_target_approximate_95_interval": [float(lo), float(hi)],
        "matched_radius_for_point_estimate": radii[0],
        "radius_range_from_target_approximate_95_interval": radii[1:],
        "achieved_gaussian_formula_cost": matched_cost,
        "old_v12_independent_nominal_estimate": previous_target,
        "radius_from_old_v12_independent_estimate": float(independent_radius),
        "old_v12_nominal_upper_with_reported_pointwise_probability": previous_nominal["policies"]["V1a"]["upper_average"],
        "old_v12_pointwise_error_probability": previous_nominal["pointwise_error_probability"],
        "frozen_points": other,
    }


def matched_replay(comparison, x0, x1, truth, onset, stop, log_bf, policies):
    metrics = {
        name: score_estimates(x1[:, j], x0[:, j], truth, log_bf, onset, stop)
        for j, name in enumerate(policies)
    }
    result = defender_comparison(metrics)
    difference = metrics[MATCHED].support_mse - metrics["V1a"].support_mse
    center = float(difference.mean())
    radius = t.ppf(.975, len(difference)-1) * difference.std(ddof=1) / np.sqrt(len(difference))
    result["matched_minus_v1a_active_mse_paired_t95_exploratory"] = {
        "n": len(difference), "mean": center, "interval": [float(center-radius), float(center+radius)],
    }
    result["sample_scope"] = comparison
    return result, metrics["V1a"]


def propagation_report(prepared, original, healthy, truth, decoded_difference, attack_trace,
                       healthy_trace, work_covariance, v1a_events):
    model, settings = prepared.model, prepared.settings
    m, horizon = prepared.m, settings.horizon
    f, gain = model.F(settings.receiver), model.G[settings.receiver]
    displacement = original - healthy
    absolute_error = original - truth
    nominal_error = healthy - truth
    fit = decoded_difference - attack_trace["delta"]
    r = fit + healthy_trace["delta"]
    prior = np.concatenate((np.zeros_like(displacement[:, :1]), displacement[:, :-1]), axis=1)
    recurrence = prior @ f.T + r @ gain.T
    np.testing.assert_allclose(displacement, recurrence, atol=2e-11, rtol=2e-11)

    cross = 2*np.sum(nominal_error*displacement, axis=-1)
    expansion = squared(nominal_error) + squared(displacement) + cross
    np.testing.assert_allclose(squared(absolute_error), expansion, atol=2e-12, rtol=2e-12)
    powers = [np.linalg.matrix_power(f, j) @ gain for j in range(horizon)]
    r_sqr = squared(r)
    e_bound = np.array([
        sum(np.sqrt(squared(r[:, t] @ powers[k-t].T).mean()) for t in range(k+1))**2
        for k in range(horizon)
    ])
    observed = squared(displacement).mean(axis=0)
    assert np.all(observed <= e_bound + 1e-12)

    output = {}
    for k in (7, 8, 9):
        ticks = np.arange(k+1)[None, :]
        alive = ticks < v1a_events.events.alarm[:, None]
        on_empirical_support = v1a_events.events.before & (v1a_events.events.danger <= k)
        risk_union = np.minimum(1., np.mean(np.sum(squared(displacement[:, :k+1])*alive, axis=1))/DANGER_SQUARED)
        assert on_empirical_support.mean() <= risk_union + 1e-12
        output[str(k-6)] = {
            "k": k,
            "decoded_distortion_fit_mse": float(squared(fit[:, k]).mean()),
            "model_posterior_cov_trace_not_assumed_calibrated": float(np.trace(work_covariance[:, k], axis1=-2, axis2=-1).mean()),
            "healthy_compensation_mean_squared": float(squared(healthy_trace["delta"][:, k]).mean()),
            "paired_residual_mean_squared": float(r_sqr[:, k].mean()),
            "paired_state_displacement_mean_squared": float(observed[k]),
            "weighted_L2_propagation_upper": float(e_bound[k]),
            "nominal_state_mse": float(squared(nominal_error[:, k]).mean()),
            "state_error_cross_term": float(cross[:, k].mean()),
            "actual_state_mse": float(squared(absolute_error[:, k]).mean()),
            "empirical_strict_pre_alarm_count_through_k": int(on_empirical_support.sum()),
            "empirical_alarm_gated_union_upper_through_k": float(risk_union),
        }
    return {
        "max_recurrence_error": float(np.max(np.abs(displacement-recurrence))),
        "max_state_mse_expansion_error": float(np.max(np.abs(squared(absolute_error)-expansion))),
        "full_horizon_displacement_mse": float(squared(displacement).mean()),
        "full_horizon_nominal_actual_mse": float(squared(nominal_error).mean()),
        "full_horizon_attacked_actual_mse": float(squared(absolute_error).mean()),
        "full_horizon_cross_term": float(cross.mean()),
        "scope": "Same 12 saved paths; empirical moment bounds only",
        "first_three": output,
    }
