"""Closed-loop Pareto screen on V1.1/V1.2 material for development only."""
import argparse
from pathlib import Path

import numpy as np

from inputs import DEFAULT_BASELINE, HERE, MODELS, V11, V12, legacy_cases, load_prepared, setup, write_json
from replay import BASE, run_batches, summarize


def compare_saved(model, size, spec, case, data):
    if size == 48:
        path = V11 / f"phase2_results/{model}_step_logs.npz"
        match = ("V1a", "V1b-unconstrained")
        ours = ("V1a", "conditional-MSE")
        indices = (1, 5)
    else:
        path = V12 / f"results/validation/{model}_paths.npz"
        match = ("V1a", "V1.2a", "V1.2a-no-cap")
        ours = ("V1a", "V1.2a", "conditional-MSE")
        indices = (1, 4, 5)
    with np.load(path, allow_pickle=False) as pool:
        for label, j, name in zip(match, indices, ours):
            a = BASE.index(name)
            np.testing.assert_allclose(data["post"][:, a], pool[f"{case}__post"][:, j],
                                       atol=2e-10, rtol=2e-10,
                                       err_msg=f"{model} {size} {case} {label}")
        if f"{case}__log_bf" in pool:
            np.testing.assert_allclose(data["log_bf"], pool[f"{case}__log_bf"], atol=5e-10)
        np.testing.assert_array_equal(data["log_bf"]>=-np.log(spec["event_false_alarm"]),
                                      pool[f"{case}__common_alarm"])
    return len(match)


def pareto_rows(rows):
    pairs = {name: (row["active_mse"][0], row["risk_before_alarm"]) for name, row in rows.items()
             if row["active_mse"] is not None}
    return [name for name, pair in pairs.items() if all(not (v[0]<=pair[0] and v[1]<=pair[1] and v!=pair)
                                                      for v in pairs.values())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    common, settings, spec, _ = setup(args.baseline)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    cohorts, checks = {}, 0
    for model in MODELS:
        p = load_prepared(args.baseline, model, common, settings)
        sets = legacy_cases(args.baseline, p, spec, common)
        model_rows = {}
        for group in sets:
            reports = {}
            size = len(group["healthy"])
            nominal = run_batches(p, group["healthy"])
            for case, streams in group.items():
                closed = nominal if case == "healthy" else run_batches(p, streams)
                checks += compare_saved(model, size, spec, case, closed)
                methods, details = summarize(p, closed, nominal, streams, BASE, spec["danger_squared"])
                reports[case] = {"methods": methods, "front": pareto_rows(methods)}
            model_rows[str(size)] = reports
        cohorts[model] = model_rows
        print(model, "archived path regression and Pareto evaluation verified", flush=True)
    write_json(out / "legacy_pareto.json", {"n_regression_method_groups": checks, "models": cohorts,
               "initial_modes": BASE, "scope": "old trajectories for screening only"})


if __name__ == "__main__":
    main()
