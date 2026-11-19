"""Defensive observation traces and post-run A0 identification scoring."""

import numpy as np
from scipy.special import expit, logit

from paired_analysis import finite_mean
from risk_metrics import first_crossing


def capture_posterior(observer, observations):
    """Capture original V1 values from receiver inputs; no evaluation labels enter."""
    model = observer.prepared
    settings = model.settings
    n, horizon = len(observations), settings.horizon
    weights = np.empty((n, horizon, horizon + 1))
    component_mean = np.zeros((n, horizon, horizon, model.m))
    delta = np.empty((n, horizon, model.m))
    log_bf = np.empty((n, horizon))
    p_active = np.empty_like(log_bf)
    p_null = logit(settings.no_event_prior)
    for k in range(horizon):
        frame = observer.at(observations, k)
        weights[:, k, 0] = expit(p_null - frame.log_bf)
        weights[:, k, 1:k + 2] = frame.posterior_weights
        weights[:, k, k + 2:] = (
            weights[:, k, :1] * (1 - settings.no_event_prior)
            / (horizon * settings.no_event_prior)
        )
        component_mean[:, k, :k + 1] = (frame.transfer @ frame.posterior_coef[..., None])[..., 0]
        delta[:, k] = frame.delta
        p_active[:, k] = frame.probability_active
        log_bf[:, k] = frame.log_bf
    np.testing.assert_allclose(weights.sum(axis=2), 1, rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.sum(weights[:, :, 1:, None] * component_mean, axis=2),
                               delta, rtol=1e-11, atol=1e-12)
    return dict(weights=weights, component_mean=component_mean, delta=delta,
                p_active=p_active, log_bf=log_bf)


def diagnose_a0(trace, healthy, decoded_truth, metrics, gain, path_ids, duration):
    """The saved onset and decoded distortion are scoring targets after replay."""
    ev = metrics.events
    n, horizon, _ = trace["delta"].shape
    starts = np.arange(horizon)
    ticks = starts[None, :]
    active = (ticks >= ev.onset[:, None]) & (ticks < ev.stop[:, None])
    can_be_active = (ticks.T >= starts) & (ticks.T < starts + duration)
    observed_weights = trace["weights"][:, :, 1:]
    map_active = np.where(can_be_active, observed_weights, -np.inf).argmax(axis=2)
    rows = np.arange(n)[:, None]
    p_truth = observed_weights[rows, ticks, ev.onset[:, None]]
    true_mean = trace["component_mean"][rows, ticks, ev.onset[:, None]]
    p_active = trace["p_active"]
    if np.any(p_active[active] <= 0):
        raise ValueError("The archived active mass must be positive for this diagnostic.")
    normalized = np.divide(trace["delta"], p_active[..., None],
                           out=np.zeros_like(trace["delta"]), where=p_active[..., None] > 0)
    p_truth_given_active = np.divide(p_truth, p_active, out=np.zeros_like(p_truth),
                                     where=active)
    components = {
        "inactive_probability_mass": trace["delta"] - normalized,
        "onset_mixture_given_active": normalized - true_mean,
        "fit_with_frozen_true_onset": true_mean - decoded_truth,
    }
    error = trace["delta"] - decoded_truth
    np.testing.assert_allclose(sum(components.values()), error, atol=1e-12)
    sse = np.sum(error ** 2, axis=-1)
    component_squares = {name: np.sum(vec ** 2, axis=-1) for name, vec in components.items()}
    total_cross = sse - sum(component_squares.values())
    mixture_gap = np.sum((trace["delta"] - true_mean) ** 2, axis=-1)
    mixture_cross = sse - mixture_gap - component_squares["fit_with_frozen_true_onset"]

    raw = np.sum((decoded_truth @ gain.T) ** 2, axis=-1)
    paired_correction = trace["delta"] - healthy["delta"]
    residual = np.sum(((decoded_truth - paired_correction) @ gain.T) ** 2, axis=-1)
    first_effect = first_crossing(active & (raw - residual > 1e-12))
    first_half = first_crossing(active & (raw > 1e-12) & (residual <= 0.5 * raw))
    first_posterior = first_crossing(active & (p_active >= 0.5))
    onset_packet = ticks == ev.onset[:, None]
    dangerous_tick = active & (ticks == ev.danger[:, None])
    phases = {
        "first_packet": onset_packet,
        "later_support": active & ~onset_packet,
        "whole_support": active,
        "first_danger_in_support": dangerous_tick,
    }
    grouped = {}
    for name, mask in phases.items():
        grouped[name] = {
            "packet_count": int(mask.sum()),
            "no_event_probability": finite_mean(trace["weights"][:, :, 0][mask]),
            "active_probability": finite_mean(p_active[mask]),
            "true_onset_probability": finite_mean(p_truth[mask]),
            "true_onset_given_active": finite_mean(p_truth_given_active[mask]),
            "active_map_correct": finite_mean((map_active == ev.onset[:, None])[mask]),
            "decoded_squared_error": finite_mean(sse[mask]),
            "mixture_gap_squared": finite_mean(mixture_gap[mask]),
            "mixture_and_true_onset_cross_term": finite_mean(mixture_cross[mask]),
            **{f"{key}_squared": finite_mean(values[mask])
               for key, values in component_squares.items()},
            "all_three_cross_terms": finite_mean(total_cross[mask]),
        }
    pre_active = ev.before & (ev.danger < ev.stop)
    summary = {
        "phases": grouped,
        "healthy_mean_no_event_probability": finite_mean(healthy["weights"][:, :, 0]),
        "paired_update_first_packet_improves": int(np.sum(first_effect == ev.onset)),
        "paired_update_first_packet_halves_residual": int(np.sum(first_half == ev.onset)),
        "paired_update_ever_improves_in_support": int(np.sum(first_effect < horizon)),
        "paired_update_ever_halves_residual_in_support": int(np.sum(first_half < horizon)),
        "strict_pre_alarm_dangers_in_support": int(pre_active.sum()),
        "strict_pre_alarm_before_first_improving_compensation": int(
            np.sum(pre_active & (ev.danger < first_effect))
        ),
        "strict_pre_alarm_before_half_residual": int(np.sum(pre_active & (ev.danger < first_half))),
        "strict_pre_alarm_before_active_probability_half": int(
            np.sum(pre_active & (ev.danger < first_posterior))
        ),
    }
    per_path = []
    for i, path_id in enumerate(path_ids):
        record = {"path_id": int(path_id), "true_onset": int(ev.onset[i])}
        for name, times in (("first_danger", ev.danger), ("first_alarm", ev.alarm),
                            ("first_improving_compensation", first_effect),
                            ("first_half_residual", first_half),
                            ("first_active_probability_at_least_half", first_posterior)):
            record[name] = int(times[i]) if times[i] < horizon else None
        record["support"] = [
            {
                "tick": int(k),
                "p_no_event": float(trace["weights"][i, k, 0]),
                "p_active": float(p_active[i, k]),
                "p_true_onset": float(p_truth[i, k]),
                "p_true_onset_given_active": float(p_truth_given_active[i, k]),
                "active_map_onset": int(map_active[i, k]),
                "decoded_squared_error": float(sse[i, k]),
                "true_onset_fit_squared_error": float(component_squares["fit_with_frozen_true_onset"][i, k]),
                "paired_update_residual_squared": float(residual[i, k]),
                "uncompensated_update_squared": float(raw[i, k]),
            }
            for k in np.flatnonzero(active[i])
        ]
        per_path.append(record)
    return summary, per_path
