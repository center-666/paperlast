"""Path-aligned defender comparisons with censoring retained."""

from itertools import combinations

import numpy as np
from scipy.stats import beta


def paired_risk(candidate, reference, confidence=0.95):
    """Bonferroni difference of two exact binomial intervals for discordant pairs."""
    candidate, reference = map(np.asarray, (candidate, reference))
    if (candidate.ndim != 1 or candidate.shape != reference.shape or not candidate.size
            or candidate.dtype != np.bool_ or reference.dtype != np.bool_
            or not 0 < confidence < 1):
        raise ValueError("Expected nonempty paired Boolean risks and a valid confidence level.")
    n = len(candidate)
    only_candidate = int(np.sum(candidate & ~reference))
    only_reference = int(np.sum(reference & ~candidate))
    tail = (1 - confidence) / 4

    def limits(count):
        lo = beta.ppf(tail, count, n - count + 1) if count else 0.0
        hi = beta.ppf(1 - tail, count + 1, n - count) if count < n else 1.0
        return float(lo), float(hi)

    cl, cu = limits(only_candidate)
    rl, ru = limits(only_reference)
    return {
        "n": n,
        "only_candidate": only_candidate,
        "only_reference": only_reference,
        "both": int(np.sum(candidate & reference)),
        "neither": int(np.sum(~candidate & ~reference)),
        "difference": (only_candidate - only_reference) / n,
        "confidence": confidence,
        "interval": [cl - ru, cu - rl],
    }


def finite_mean(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return float(valid.mean()) if valid.size else None


def defender_comparison(methods):
    summary = {}
    for name, row in methods.items():
        n = len(row.events.danger)
        counts = row.events.counts()
        summary[name] = {
            "n": n,
            "event_counts": counts,
            "strict_pre_alarm_rate": counts["before_alarm"] / n,
            "first_packet_rate": counts["first_packet"] / n,
            "healthy_mse": finite_mean(row.healthy_mse),
            "support_mse": finite_mean(row.support_mse),
            "tail_mse": finite_mean(row.tail_mse),
        }
        if row.full_mse is not None:
            summary[name]["full_mse"] = finite_mean(row.full_mse)

    pairs = {}
    for reference, candidate in combinations(methods, 2):
        a, b = methods[candidate], methods[reference]
        np.testing.assert_array_equal(a.events.alarm, b.events.alarm)
        np.testing.assert_array_equal(a.events.onset, b.events.onset)
        np.testing.assert_array_equal(a.events.stop, b.events.stop)
        pairs[f"{candidate} - {reference}"] = {
            "before_alarm": paired_risk(a.events.before, b.events.before),
            "first_packet": paired_risk(a.events.first_packet, b.events.first_packet),
            "support_mse_difference": finite_mean(a.support_mse - b.support_mse),
            "tail_mse_difference": finite_mean(a.tail_mse - b.tail_mse),
            "healthy_mse_difference": finite_mean(a.healthy_mse - b.healthy_mse),
        }
    return {"methods": summary, "paired_differences": pairs}


def v1a_tradeoff(v1a, saturation, path_ids):
    """Identify which archived paths favor either receiver under both criteria."""
    ids = np.asarray(path_ids)
    diff = v1a.support_mse - saturation.support_mse
    if ids.shape != diff.shape or not np.all(np.isfinite(diff)):
        raise ValueError("Expected identities and active MSE from the same paired trajectories.")
    v_risk, s_risk = v1a.events.before, saturation.events.before
    v_safer, s_safer = ~v_risk & s_risk, v_risk & ~s_risk
    v_accurate, s_accurate = diff < -1e-12, diff > 1e-12
    groups = {
        "v1a_only_avoids_pre_alarm_danger": v_safer,
        "saturation_only_avoids_pre_alarm_danger": s_safer,
        "v1a_lower_support_mse": v_accurate,
        "saturation_lower_support_mse": s_accurate,
        "v1a_pareto_improves": (~s_safer & ~s_accurate) & (v_safer | v_accurate),
        "saturation_pareto_improves": (~v_safer & ~v_accurate) & (s_safer | s_accurate),
        "v1a_lower_mse_but_saturation_safer": v_accurate & s_safer,
        "saturation_lower_mse_but_v1a_safer": s_accurate & v_safer,
    }
    return {key: {"count": int(mask.sum()), "path_ids": ids[mask].tolist()}
            for key, mask in groups.items()}
