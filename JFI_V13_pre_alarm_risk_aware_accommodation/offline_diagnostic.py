"""One-step reachability audit using truth only for retrospective comparisons."""
import argparse
from pathlib import Path

import numpy as np

from inputs import DEFAULT_BASELINE, HERE, MODELS, legacy_cases, load_prepared, setup, write_json
from replay import first, run_batches


def audit(p, measurements, nominal, d2):
    run = run_batches(p, measurements, ("V1a",))
    g = p.model.G[p.settings.receiver]
    displacement = run["post"][:,0]-nominal["post"][:,0]
    direction = run["delta"] @ g.T
    baseline = displacement + direction
    s = np.sum(direction**2,axis=-1)
    q = np.sum(baseline*direction,axis=-1)
    best = np.divide(q,s,out=np.ones_like(s),where=s>1e-24).clip(0,1)
    attainable = np.sum((baseline-best[...,None]*direction)**2,axis=-1)
    power = np.sum(displacement**2,axis=-1)
    first_hit = first(power>=d2)
    first_alarm = first(run["log_bf"]>=-np.log(p.settings.event_false_alarm))
    selected = np.flatnonzero(first_hit<first_alarm)
    first_packet_power = power[np.arange(len(measurements)),measurements.start]
    min_at_first = attainable[np.arange(len(measurements)),measurements.start]
    return {"v1a_pre_alarm_first_hits":int(len(selected)),
            "cannot_remove_with_scalar_at_that_same_history":int(np.sum(attainable[selected,first_hit[selected]]>=d2)),
            "first_packet_crossings":int(np.sum(first_packet_power>=d2)),
            "first_packet_crossings_unreachable_by_reducing_weight":int(np.sum(min_at_first>=d2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    common, settings, spec, _ = setup(args.baseline)
    rows = {}
    for model in MODELS:
        p = load_prepared(args.baseline,model,common,settings)
        _, tests = legacy_cases(args.baseline,p,spec,common)
        nominal = run_batches(p,tests["healthy"],("V1a",))
        rows[model] = {key:audit(p,tests[key],nominal,spec["danger_squared"])
                       for key in ("A0_onset_replay","A1_H_mismatch","A1_C_mismatch")}
    write_json(HERE/"results/offline_action_reachability.json",
               {"scope":"Archived 96-path trajectories; fixed V1a past and current healthy output.",
                "oracle":"Retrospective single corrupted-side weight only. Not an online policy or global bound.",
                "models":rows})


if __name__=="__main__":
    main()
