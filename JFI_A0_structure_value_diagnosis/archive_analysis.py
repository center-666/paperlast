"""Compare frozen operating points using path-aligned saved A0 outcomes."""

import numpy as np
from scipy.stats import t

from paired_analysis import paired_risk
from risk_metrics import DANGER_SQUARED, events_from_times


SATURATION = ("anchor@0.1%", "anchor@0.5%", "anchor@1%", "anchor@2%")
POLICIES = ("V0", "V1a", "V1.2a", "V1.4", *SATURATION)
CONFIDENCE = 0.95
FAMILY_SIZE = 2 * len(SATURATION)


def mean_interval(differences, confidence=CONFIDENCE):
    """Student-t interval on independent path-level differences."""
    values = np.asarray(differences)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("A paired interval needs at least two finite path differences.")
    center = values.mean()
    radius = t.ppf((1 + confidence) / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(center), "interval": [float(center - radius), float(center + radius)]}


def nominal_health(conditional, index, base_index, v0_exact, old):
    # Subtract conditional losses on the same paths and add the analytic V0 cost.
    paired = mean_interval((conditional[:, index] - conditional[:, base_index]) * (100 / v0_exact))
    pct = paired["mean"]
    np.testing.assert_allclose(
        [pct, *paired["interval"]], old["health_expectation_pct_conditional_MC95"],
        rtol=1e-9, atol=2e-10,
    )
    formula = old.get("health_expected_MSE_exact_formula")
    return {
        "method": "Gaussian model formula" if formula is not None else "conditional Monte Carlo vs exact V0",
        "mse": formula if formula is not None else v0_exact * (1 + pct / 100),
        "relative_pct": old["health_expected_pct_exact_formula"] if formula is not None else pct,
        "relative_pct_conditional_estimate": paired,
        "nominal_penalty_upper_pct_if_proved": old.get("analytic_nominal_health_penalty_upper_pct"),
    }


def analyze_archive(path, model_receipt, fixed_onset, horizon):
    table = {spec["name"]: (i, spec) for i, spec in enumerate(model_receipt["method_table"])}
    old_health = model_receipt["cases"]["healthy"]["methods"]
    v0_exact = old_health["V0"]["health_expected_MSE_exact_formula"]
    prefix = "unknown_start__"
    with np.load(path, allow_pickle=False) as data:
        all_onsets = data[prefix + "onset"]
        keep = all_onsets == fixed_onset
        onset = all_onsets[keep]
        stop = data[prefix + "stop"][keep]
        alarm = data[prefix + "first_alarm"][keep]
        danger = data[prefix + "first_danger"][keep]
        first = data[prefix + "first_packet"][keep]
        strict = data[prefix + "pre_alarm"][keep]
        all_active = data[prefix + "active_mean_mse"]
        active = all_active[keep]
        peak = data[prefix + "peak_paired_squared"][keep]
        healthy = data["healthy__health_mean_mse"]
        conditional = data["healthy__conditional_health_squared"]
        np.testing.assert_array_equal(data[prefix + "health_mean_mse"][keep], healthy[keep])

    np.testing.assert_array_equal(stop, onset + 10)
    v1a_index = table["V1a"][0]
    rows = {}
    comparisons = {}
    for name in POLICIES:
        col, config = table[name]
        events = events_from_times(danger[:, col], alarm, onset, stop, horizon)
        np.testing.assert_array_equal(events.first_packet, first[:, col])
        np.testing.assert_array_equal(events.before, strict[:, col])
        np.testing.assert_array_equal(events.observed, peak[:, col] >= DANGER_SQUARED)
        counts = events.counts()
        rows[name] = {
            "frozen_method": config,
            "nominal_health": nominal_health(conditional, col, table["V0"][0], v0_exact, old_health[name]),
            "sample_health_mse": float(healthy[keep, col].mean()),
            "all_healthy_archive_mse": float(healthy[:, col].mean()),
            "active_mse": float(active[:, col].mean()),
            "events": counts,
            "strict_pre_alarm_rate": counts["before_alarm"] / len(onset),
            "first_packet_rate": counts["first_packet"] / len(onset),
        }
        if name in SATURATION:
            difference = active[:, col] - active[:, v1a_index]
            comparisons[name] = {
                "active_mse_sat_minus_v1a_pointwise_t95": mean_interval(difference),
                "active_mse_sat_minus_v1a_family_t95": mean_interval(
                    difference, confidence=1 - (1 - CONFIDENCE) / FAMILY_SIZE
                ),
                "v1a_lower_mse_paths": int(np.sum(difference > 1e-12)),
                "saturation_lower_mse_paths": int(np.sum(difference < -1e-12)),
                "median_mse_difference": float(np.median(difference)),
                "strict_pre_alarm_sat_minus_v1a": paired_risk(events.before, strict[:, v1a_index]),
                "first_packet_sat_minus_v1a": paired_risk(events.first_packet, first[:, v1a_index]),
            }
    sensitivity = {}
    for other in (3, 11):
        subset = all_active[all_onsets == other]
        sensitivity[str(other)] = {
            "path_count": len(subset),
            "active_mse_sat_minus_v1a_pointwise_t95": {
                name: mean_interval(subset[:, table[name][0]] - subset[:, v1a_index])
                for name in SATURATION
            },
        }
    return {
        "provenance": "saved held-out A0 outcomes (receiver inputs unavailable for rerun)",
        "onset": fixed_onset,
        "stop_exclusive": fixed_onset + 10,
        "path_count": len(onset),
        "healthy_counts_for_nominal_estimation": len(healthy),
        "mse_family_comparisons": FAMILY_SIZE,
        "methods": rows,
        "paired_sat_minus_v1a": comparisons,
        "other_onsets_descriptive_sensitivity": sensitivity,
    }
