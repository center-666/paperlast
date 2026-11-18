"""Pre-registered archived screen, plus an optional gated independent replay."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from v14_inputs import (DEFAULT_BASELINE, HERE, MODELS, PRIMARY, V11, V12, digest,
                        fresh_streams, joint_cases, legacy_cases, load_json, load_prepared, setup,
                        write_json)
from dual_receiver import DESIGN, METHODS
from paired_replay import collect, evaluate


def check_spec(settings, spec):
    assert set(MODELS)==set(DESIGN["models"])
    assert settings.horizon==DESIGN["horizon"]==spec["horizon"]
    assert settings.nominal_one_step_allowance==DESIGN["nominal_allocation_fraction"]
    assert spec["danger_squared"]==DESIGN["danger_squared"]
    assert settings.event_false_alarm==.05
    assert DESIGN["development"]["primary"]==list(PRIMARY)


def previous_paths(model, size, case, replay):
    saved = (V11/f"phase2_results/{model}_step_logs.npz" if size==48
             else V12/f"results/validation/{model}_paths.npz")
    tests = (("V1a",1),) if size==48 else (("V1a",1),("V1.2a",4))
    with np.load(saved,allow_pickle=False) as old:
        for name,j in tests:
            np.testing.assert_allclose(replay["post"][:,METHODS.index(name)],old[case+"__post"][:,j],
                                       atol=2e-10,rtol=3e-10,err_msg=f"{model} {size} {case} {name}")
        np.testing.assert_array_equal(replay["log_bf"]>=-np.log(.05),old[case+"__common_alarm"])
    return len(tests)


def run_group(p, spec, groups, paths, model, size=None):
    result = {}
    baseline = collect(p,groups["healthy"])
    checks = 0
    for case,stream in groups.items():
        replay = baseline if case=="healthy" else collect(p,stream)
        if size in (48,96):
            checks += previous_paths(model,size,case,replay)
        outcome, sample = evaluate(p,stream,baseline,replay,spec["danger_squared"])
        # Archived topic paths are batch-matched with the same nominal input and watermark.
        for field in sample:
            paths[f"{case}__{field}"] = sample[field]
        result[case] = outcome
    return result,checks


def development_gate(models):
    geometry = load_json(HERE/"results/geometry.json")
    gain = sum(row["triangle_avoided"]-row["structural_avoided"]
               for model in geometry["models"].values() for row in model.values())
    health = {m:models[m]["96"]["healthy"]["methods"]["V1.4"]["health_relative_vs_V0"] for m in MODELS}
    diffs = []
    tol = DESIGN["development"]["max_relative_active_mse_vs_V12a"]
    for model in MODELS:
        for case in PRIMARY:
            methods = models[model]["96"][case]["methods"]
            v, s, t = [methods[name] for name in ("V1.4","V1.2a","trusted-budget")]
            diffs.append({"model":model,"profile":case,
                          "count_difference_vs_V12a":v["pre_alarm_danger"]-s["pre_alarm_danger"],
                          "mse_relative_vs_V12a":v["active_mse"][0]/s["active_mse"][0]-1,
                          "count_difference_vs_trusted_budget":v["pre_alarm_danger"]-t["pre_alarm_danger"],
                          "mse_relative_vs_trusted_budget":v["active_mse"][0]/t["active_mse"][0]-1})
    component = {
        "geometry_gain_vs_structural":gain>0,
        "health_ensemble_within_2pct":all(r<=DESIGN["nominal_allocation_fraction"] for r in health.values()),
        "risk_at_least_one_improvement_vs_V12a":any(r["count_difference_vs_V12a"]<0 for r in diffs),
        "risk_no_increases_vs_V12a":all(r["count_difference_vs_V12a"]<=0 for r in diffs),
        "active_mse_margin_vs_V12a":all(r["mse_relative_vs_V12a"]<=tol for r in diffs),
        "risk_and_mse_gain_vs_trusted_budget":any(r["count_difference_vs_trusted_budget"]<0 and
                                                r["mse_relative_vs_trusted_budget"]<=tol for r in diffs)
    }
    return {"passed":all(component.values()),"components":component,"primary_health":health,
            "differences":diffs,"new_physical_paths_generated_in_development":0}


def inputs_manifest(paths):
    return {str(path.relative_to(HERE)).replace("\\","/"):digest(path) for path in sorted(paths)}


def run(base, phase):
    common, settings, spec, _ = setup(base)
    check_spec(settings,spec)
    started = perf_counter()
    stage = HERE/"results"/phase
    code = inputs_manifest([*HERE.glob("*.py"), HERE/"registration.json"])
    if phase=="holdout":
        dev = load_json(HERE/"results/development/summary.json")
        if not dev["gate"]["passed"]:
            raise SystemExit("Archived development gate failed. No independent paths are drawn.")
        if code!=dev["source_hashes"]:
            raise SystemExit("The code hash differs from the gated development run.")
        stage.mkdir(parents=True,exist_ok=True)
        write_json(stage/"freeze.json",{"development_sha256":digest(HERE/"results/development/summary.json"),
                                       "source_hashes":code,"registration":DESIGN["conditional_holdout"]})
    else:
        stage.mkdir(parents=True,exist_ok=True)
    models, checks = {}, 0
    for model in MODELS:
        p = load_prepared(base,model,common,settings)
        if phase=="development":
            old48,old96 = legacy_cases(base,p,spec,common)
            groups = ((48,old48),(96,old96),(32,joint_cases(base,p,spec,common)))
        else:
            hold = DESIGN["conditional_holdout"]
            count = hold["paths_per_model_and_profile"]
            groups = ((count,fresh_streams(base,p,spec,hold["seeds"][model],count)),)
        out = {}
        for size,group in groups:
            arrays = {}
            out[str(size)], n = run_group(p,spec,group,arrays,model,size if phase=="development" else None)
            checks += n
            np.savez_compressed(stage/f"{model}_{size}_paths.npz",**arrays)
            print(model,size," ".join(f'{case}={row["methods"]["V1.4"]["pre_alarm_danger"]}'
                                     for case,row in out[str(size)].items()),flush=True)
        models[model] = out
    info = {"phase":phase,"methods":list(METHODS),"models":models,"source_hashes":code,
            "wall_seconds":perf_counter()-started,"baseline_post_regressions":checks,
            "evaluation":"All recorded impairments are independent of receiver output; the matched-history "
                         "archives preserve their original observation pairing."}
    if phase=="development":
        info["gate"] = development_gate(models)
        print(info["gate"]["components"],flush=True)
    write_json(stage/"summary.json",info)


if __name__=="__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    arg.add_argument("--phase",choices=("development","holdout"),default="development")
    opts = arg.parse_args()
    run(opts.baseline,opts.phase)
