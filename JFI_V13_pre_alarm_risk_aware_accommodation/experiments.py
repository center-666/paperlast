"""Freeze using development only, then evaluate independent receiver replays."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t

from inputs import (DEFAULT_BASELINE, FRESH, HERE, MODELS, V12, digest, fresh_streams,
                    joint_cases, load_json, load_prepared, setup, write_json)
from nominal_audit import bounded_mean_upper, rational_envelope
from replay import interval, paired_risk, run_batches, summarize
from risk_policy import make_risk

REPORT_BASE = ("V1a", "V1.2a", "conditional-MSE")
PRIMARY = ("unknown_start", "A1_H_replay", "A1_C_replay")
EDGES = np.array([0., .005, .02, .05, .1, .2, .4, .6, .8, 1.])


def verify_design(design):
    for name, expected in design["sources"].items():
        if digest(HERE/name) != expected:
            raise ValueError(f"Registered source differs: {name}")
    old = load_json(V12/"registration.json")
    prior = {int(s) for streams in old["samples"].values() for key in ("physical", "watermark")
             for s in streams[key]}
    seeds = [v for model in design["seeds"].values() for v in model.values()]
    if len(set(seeds)) != len(seeds) or prior.intersection(seeds):
        raise ValueError("Some independent seed IDs overlap.")


def calibration(details, methods, danger):
    scores = details["risk_score"]
    observed = details["offset_sq"]>=danger
    first = np.arange(scores.shape[-1])[None, :] == details["start"][:, None]
    output = {}
    for j, name in enumerate(methods):
        p, z = scores[:, j], observed[:, j]
        valid = np.isfinite(p)
        if not valid.any():
            continue
        p = np.clip(p, 0., 1.)
        bins = np.minimum(np.searchsorted(EDGES, p, side="right")-1, len(EDGES)-2)
        cs = []
        for b in range(len(EDGES)-1):
            keep = valid & (bins==b)
            n = keep.sum(axis=1)
            if n.sum()>0:
                actual = (z & keep).sum(axis=1)
                rate = float(actual.sum()/n.sum())
                pred = float(p[keep].mean())
                resid = actual-rate*n
                se = float(resid.std(ddof=1)/(np.sqrt(len(resid))*n.mean()))
                cs.append({"range": EDGES[b:b+2].tolist(), "packets": int(n.sum()),
                           "episodes": int(np.count_nonzero(n)), "predicted": pred,
                           "observed": rate,
                           "episode_cluster_interval95": [max(0.,rate-1.96*se), min(1.,rate+1.96*se)]})
        e = valid & first
        output[name] = {"brier": float(np.mean((p[valid]-z[valid])**2)),
                        "ece": float(sum(abs(r["predicted"]-r["observed"])*r["packets"] for r in cs)/valid.sum()),
                        "observed": float(z[valid].mean()), "predicted": float(p[valid].mean()),
                        "first_packet_predicted": float(p[e].mean()) if e.any() else None,
                        "first_packet_observed": float(z[e].mean()) if e.any() else None,
                        "first_eligible_n": int(e.sum()), "bins": cs}
    return output


def evaluate(base, common, settings, spec, design, phase, lambdas, save_paths):
    folder = HERE/"results"/phase
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    methods = REPORT_BASE+tuple(lambdas)
    risk = make_risk(lambdas, spec["danger_squared"], design["angular_nodes"])
    n = design["paths"][phase]
    for model in MODELS:
        p = load_prepared(base, model, common, settings)
        if phase == "historical_joint":
            cases = joint_cases(base, p, spec, common)
        else:
            cases = fresh_streams(base, p, spec, design["seeds"][model][phase], n)
        active_clock = perf_counter()
        clean = run_batches(p, cases["healthy"], methods, risk, design["batch"])
        rows = {}
        packed = {}
        for case, streams in cases.items():
            data = clean if case=="healthy" else run_batches(p, streams, methods, risk, design["batch"])
            report, raw = summarize(p, data, clean, streams, methods, spec["danger_squared"])
            summary = {"n":len(streams),"methods":report,
                       "calibration":calibration(raw, methods, spec["danger_squared"])}
            if case != "healthy":
                for j, label in enumerate(methods):
                    gap = raw["mse"][:, j]-(1+design["mse_screen_relative_margin"])*raw["mse"][:, 0]
                    upper = gap.mean()+t.ppf(.95, len(gap)-1)*gap.std(ddof=1)/np.sqrt(len(gap))
                    if label in lambdas:
                        method = report[label]
                        method["mse_1pct_margin_point"] = float(gap.mean())
                        method["mse_1pct_margin_upper95_t"] = float(upper)
                        method["pre_alarm_simultaneous_upper95"] = paired_risk(
                            raw["risk"][:, j], raw["risk"][:, 0], tail=.05/(2*len(PRIMARY)))["exact_interval"][1]
            rows[case] = summary
            if save_paths:
                for field,val in raw.items():
                    if val.ndim==3 and val.dtype==np.float64:
                        val = val.astype(np.float32)
                    packed[case+"__"+field] = val
            print(phase, model, case, [(key,report[key]["risk_before_alarm"],
                  None if report[key]["active_mse"] is None else round(report[key]["active_mse"][0],5))
                  for key in ("V1a",*lambdas)], flush=True)
        if save_paths:
            np.savez_compressed(folder/f"{model}_paths.npz", **packed)
        out[model] = {"profiles":rows,"seconds_batch":perf_counter()-active_clock}
    return {"methods":methods, "angles":design["angular_nodes"], "profile_scope":phase,
            "design_sha256":digest(HERE/"design.json"), "models":out}


def choose(development, candidates, design):
    rows = {}
    for label, penalty in candidates.items():
        mse = []
        relative_health = []
        risk_change = []
        abs_deviation = []
        for model in MODELS:
            profiles = development["models"][model]["profiles"]
            nominal = profiles["healthy"]["methods"]
            relative_health.append(nominal[label]["healthy_mse"][0]/nominal["V1a"]["healthy_mse"][0]-1)
            for key in FRESH[1:]:
                r = profiles[key]["methods"][label]
                mse.append(r["mse_relative"])
                if key in PRIMARY:
                    risk_change.append(r["paired_risk"]["difference"])
                abs_deviation.append(1-r["phases"]["active_later"]["mean_alpha"])
        rows[label] = {"lambda":penalty, "max_support_relative_mse":max(mse),
                       "nominal_relative_to_V1a_development":relative_health,
                       "mean_primary_risk_difference":float(np.mean(risk_change)),
                       "max_primary_risk_difference":max(risk_change),
                       "mean_weight_shortfall":float(np.mean(abs_deviation)),
                       "within_development_mse_screen":max(mse)<=design["mse_screen_relative_margin"]}
    eligible = [key for key,r in rows.items() if r["within_development_mse_screen"]]
    order = lambda name: (rows[name]["mean_primary_risk_difference"],rows[name]["max_primary_risk_difference"],
                          rows[name]["mean_weight_shortfall"],-rows[name]["lambda"])
    picked = min(eligible,key=order) if eligible else max(candidates,key=lambda key:candidates[key])
    return {"selected":picked, "lambda":candidates[picked], "passed_development_mse_screen":bool(eligible),
            "scores":rows,"development_sha256":digest(HERE/"results/development/summary.json"),
            "design_sha256":digest(HERE/"design.json")}


def healthy_check(base, common, settings, spec, design, coeff):
    """Independent conditional healthy-MSE integration using the V1.2 envelope."""
    out = HERE/"results"/"healthy_check"
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    n = design["paths"]["healthy_check"]
    risk = make_risk({"V1.3":coeff},spec["danger_squared"],design["angular_nodes"])
    methods = ("V1a","V1.3")
    for model in MODELS:
        p = load_prepared(base,model,common,settings)
        streams = fresh_streams(base,p,spec,design["seeds"][model]["healthy_check"],n,
                                healthy_only=True)["healthy"]
        observed = run_batches(p,streams,methods,risk,design["batch"])
        exact, trace = rational_envelope(base/"common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json",
                                        model,spec)
        np.testing.assert_allclose(p.nominal_mse(), exact, atol=1e-12)
        conditional = observed["null_score"].mean(axis=2)
        fits = np.sum((observed["post"]-streams.state[:,None])**2,axis=-1).mean(axis=2)
        clip = 2.65
        remainder = .2*settings.horizon*(settings.template_duration+1)*np.exp(-clip/.2)
        upper = bounded_mean_upper(conditional,clip,.05/4)+remainder+1e-9+trace.mean()
        nominal = float(exact.mean())
        table[model] = {"exact_v0_mse":nominal, "n":n, "tail_allowance":float(remainder),
                        "policies": {key:{"mean_sample_and_t_interval":interval(fits[:,j]),
                            "model_mean_estimate":float(trace.mean()+conditional[:,j].mean()),
                            "upper_one_sided_joint95":float(upper[j]),
                            "upper_relative_to_exact_v0":float(upper[j]/nominal-1)}
                            for j,key in enumerate(methods)}}
        np.savez_compressed(out/f"{model}_integrands.npz",
                            conditional=conditional, paired_mse=fits, method_labels=methods)
        print("nominal",model,"V1.3 upper",float(upper[-1]/nominal-1),flush=True)
    write_json(out/"summary.json",table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    parser.add_argument("--phase",choices=("development","validation","healthy_check","historical_joint"),required=True)
    args = parser.parse_args()
    design = load_json(HERE/"design.json")
    verify_design(design)
    common, settings, spec, _ = setup(args.baseline)
    phase = args.phase
    if phase == "development":
        if (HERE/"frozen_policy.json").is_file():
            raise FileExistsError("The saved decision is already frozen. Use an isolated copy for refitting.")
        candidates = {f"risk-{x:g}":x for x in design["lambda_grid"]}
        summary = evaluate(args.baseline,common,settings,spec,design,phase,candidates,False)
        write_json(HERE/"results/development/summary.json",summary)
        frozen = choose(summary,candidates,design)
        with (HERE/"frozen_policy.json").open("x",encoding="utf-8",newline="\n") as file:
            import json
            json.dump(frozen,file,ensure_ascii=False,indent=2)
            file.write("\n")
        print("Frozen lambda",frozen["lambda"],flush=True)
        return
    frozen = load_json(HERE/"frozen_policy.json")
    if frozen["design_sha256"]!=digest(HERE/"design.json") or frozen["development_sha256"]!=digest(HERE/"results/development/summary.json"):
        raise ValueError("The design or tuning results differ from the freeze.")
    if phase == "healthy_check":
        healthy_check(args.baseline,common,settings,spec,design,frozen["lambda"])
    else:
        result = evaluate(args.baseline,common,settings,spec,design,phase,{"V1.3":frozen["lambda"]},phase=="validation")
        result["frozen_policy_sha256"] = digest(HERE/"frozen_policy.json")
        write_json(HERE/"results"/phase/"summary.json",result)


if __name__ == "__main__":
    main()
