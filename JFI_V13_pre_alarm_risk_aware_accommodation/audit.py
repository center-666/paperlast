"""Numerical integration and per-packet runtime diagnostics on archived input."""
import argparse
import gc
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from inputs import DEFAULT_BASELINE, HERE, MODELS, legacy_cases, load_json, load_prepared, setup, write_json
from replay import play
from risk_policy import OnlineRiskReceiver, make_risk
from accommodation import OnlineSoftCompensator


def measure(factory, streams, m, count):
    def packets(i):
        return zip(streams.trusted[i, :, :m], streams.trusted[i, :, m:],
                   streams.target[i], streams.decoder[i])

    calibration = factory()
    for packet in packets(0):
        calibration.step(*packet)
    ticks = []
    for i in range(count):
        update = factory().step
        for packet in packets(i):
            start = perf_counter()
            update(*packet)
            ticks.append(perf_counter()-start)
    gc.collect()
    tracemalloc.start()
    try:
        update = factory().step
        for packet in packets(0):
            update(*packet)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    values = np.array(ticks)*1000
    return {"steps":len(ticks),"milliseconds_mean":float(values.mean()),
            "milliseconds_q50_q90_q99":np.quantile(values,[.5,.9,.99]).tolist(),
            "python_and_traced_numpy_peak_bytes":int(peak)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    common, settings, spec, _ = setup(args.baseline)
    reg = load_json(HERE/"design.json")
    frozen = load_json(HERE/"frozen_policy.json")
    d = spec["danger_squared"]
    reports = {}
    for model in MODELS:
        p = load_prepared(args.baseline,model,common,settings)
        examples = legacy_cases(args.baseline,p,spec,common)[0]
        rows, t = {}, {}
        for label in ("healthy","A0","A1_C_mismatch"):
            test = examples[label].take(slice(0,12))
            a = play(p,test,("V1a",),make_risk({},d,reg["angular_nodes"]))["predicted"]
            b = play(p,test,("V1a",),make_risk({},d,128))["predicted"]
            delta = np.abs(a-b)
            delta = delta[np.isfinite(delta)]
            rows[label] = {"pre_alarm_points":len(delta),"mean_abs":float(delta.mean()),
                           "q95_abs":float(np.quantile(delta,.95)),"maximum_abs":float(delta.max())}
            if label in ("healthy","A1_C_mismatch"):
                old = lambda: OnlineSoftCompensator(p,common.common_step,full_compensation=True)
                new = lambda: OnlineRiskReceiver(p,common.common_step,frozen["lambda"],d,reg["angular_nodes"])
                t[label] = {"V1a":measure(old,test,p.m,8),"V1.3":measure(new,test,p.m,8)}
                times = {}
                for key, proxy in (("common_backend_full",None),
                                   ("common_backend_with_risk",make_risk({"V1.3":frozen["lambda"]},d,reg["angular_nodes"]))):
                    labs = ("V1a",) if proxy is None else ("V1.3",)
                    elapsed = []
                    for _ in range(4):
                        tick = perf_counter()
                        play(p,test,labs,proxy)
                        elapsed.append(perf_counter()-tick)
                    times[key+"_median_ms_per_update"] = float(np.median(elapsed)*1000/(len(test)*settings.horizon))
                t[label]["matched_backend_throughput"] = times
        reports[model] = {"quadrature_fixed_action_16_vs_128":rows,"latency_after_setup":t}
    write_json(HERE/"results/numerical_and_runtime_audit.json",reports)


if __name__=="__main__":
    main()
