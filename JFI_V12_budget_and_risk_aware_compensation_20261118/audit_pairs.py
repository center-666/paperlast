"""Evaluate saved, already paired receiver paths without reselection of inputs."""
import argparse

import numpy as np

from shared import DEFAULT_BASELINE, HERE, V11, load_json, load_prepared, setup, write_json
from receiver_experiment import METHODS, paired_event
from weight_experiment import first_flags


def legacy_crossings(path, p, spec, methods):
    a, b = methods.index("V1a"), methods.index("V1.1")
    f = p.model.F(p.settings.receiver)
    g = p.model.G[p.settings.receiver]
    threshold = spec["danger_squared"]
    outputs = {}
    with np.load(path, allow_pickle=False) as z:
        clean = z["healthy__post"]
        profiles = sorted({key.split("__", 1)[0] for key in z.files}-{"healthy"})
        for case in profiles:
            get = lambda field: z[f"{case}__{field}"]
            post = get("post")
            shift = post-clean
            sq = np.sum(shift**2, axis=-1)
            alarm = first_flags(get("common_alarm"))
            danger = first_flags(sq >= threshold)
            events = danger < alarm[:, None]
            trials = np.flatnonzero(events[:, b] & ~events[:, a])
            rows = []
            for i in trials:
                k = int(danger[i, b])
                difference = shift[i, b, k]-shift[i, a, k]
                previous = f @ (shift[i, b, k-1]-shift[i, a, k-1]) if k else np.zeros(p.n)
                correction = get("correction")
                healthy_correction = z["healthy__correction"]
                direct = g @ ((correction[i, a, k]-correction[i, b, k]) -
                              (healthy_correction[i, a, k]-healthy_correction[i, b, k]))
                np.testing.assert_allclose(previous+direct, difference, atol=3e-12)
                row = {"paired_index_zero_based": int(i), "first_danger_V11": k,
                       "first_danger_V1a": int(danger[i, a]), "first_shared_alarm": int(alarm[i]),
                       "V1a_power_same_packet": float(sq[i, a, k]), "V11_power": float(sq[i, b, k]),
                       "V11_alpha": float(get("alpha")[i, b, k]),
                       "V11_cap_binding": bool(get("cap_binds")[i, b, k]),
                       "V11_proposal": float(get("posterior_clipped_weight")[i, b, k]),
                       "V11_historical_difference_norm": float(np.linalg.norm(previous)),
                       "V11_current_difference_norm": float(np.linalg.norm(direct)),
                       "power_difference_from_current_correction_given_history": float(
                            sq[i, b, k]-np.linalg.norm(shift[i, a, k]+previous)**2)}
                rows.append(row)
            outputs[case] = {"count_V1a": int(events[:, a].sum()), "count_V11": int(events[:, b].sum()),
                             "comparison": paired_event(events[:, b], events[:, a], True),
                             "only_V11_danger_rows": rows}
    return outputs


def audit_validation(path, spec, p):
    a, b = METHODS.index("V1a"), METHODS.index("V1.2a")
    f = p.model.F(p.settings.receiver)
    threshold = spec["danger_squared"]
    reports = {}
    with np.load(path, allow_pickle=False) as z:
        clean = z["healthy__post"]
        profiles = sorted({name.split("__", 1)[0] for name in z.files}-{"healthy"})
        for profile in profiles:
            get = lambda field: z[f"{profile}__{field}"]
            times = get("onset")
            risk = get("before_alarm")
            damage = get("displacement_squared")
            peak = get("peak_paired_displacement_squared")
            i = np.arange(len(peak))
            same_packet_diff = get("post")[i, a, times]-clean[i, a, times]
            correction = get("correction")[i, a, times] - z["healthy__correction"][i, a, times]
            reference = get("post")[i, 0, times]-clean[i, 0, times]
            np.testing.assert_allclose(same_packet_diff, reference-correction @ p.model.G[p.settings.receiver].T,
                                       atol=4e-12)
            offset = get("post")[:, a]-clean[:, a]
            v0 = get("post")[:, 0]-clean[:, 0]
            added = offset-v0
            exc = np.flatnonzero(risk[:, a] & ~risk[:, 0])
            rows = []
            for index in exc:
                k = int(get("first_danger")[index, a])
                state_payload = added[index, k]
                lag = added[index, k-1] @ f.T if k else np.zeros(p.n)
                rows.append({"paired_index_zero_based": int(index), "first_crossing": k,
                             "first_shared_alarm": int(get("first_alarm_common")[index]),
                             "V0_at_crossing_power": float(damage[index, 0, k]),
                             "V1a_crossing_power": float(damage[index, a, k]),
                             "V1a_effect_of_relative_correction_norm": float(np.linalg.norm(state_payload)),
                             "effect_from_current_only_norm": float(np.linalg.norm(state_payload-lag)),
                             "at_first_packet": bool(k == times[index])})
            reports[profile] = {
                "V1a_first_packet_reduced_power_vs_V0": int(np.sum(damage[i, a, times] < damage[i, 0, times])),
                "V1a_first_packet_increased_power_vs_V0": int(np.sum(damage[i, a, times] > damage[i, 0, times])),
                "V1a_only_danger_before_alarm_vs_V0": rows,
                "V1a_only_first_packet_danger_vs_V0": int(np.sum((damage[i, a, times] >= threshold) &
                                                              (damage[i, 0, times] < threshold))),
                "V1a_minus_V12a_peak_bias_mean": float(np.mean(peak[:, a]-peak[:, b])),
                "paired_V1a_vs_V0": paired_event(risk[:, a], risk[:, 0], True)}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, type=type(DEFAULT_BASELINE))
    args = parser.parse_args()
    common, settings, spec, _ = setup(args.baseline)
    old = load_json(V11 / "phase2_results/heldout.json")
    outputs = {}
    for name in old["models"]:
        p = load_prepared(args.baseline, name, common, settings)
        outputs[name] = {"old_48": legacy_crossings(V11 / f"phase2_results/{name}_step_logs.npz", p, spec,
                                                  old["policies"]),
                         "new_96": audit_validation(HERE / f"results/validation/{name}_paths.npz", spec, p)}
    write_json(HERE / "results/paired_risk_audit.json", outputs)


if __name__ == "__main__":
    main()
