"""Validate the frozen inputs and reproduce the health cost and offline risk audit."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np

from archived_statistics import archive_events, health_interval, matching_interval, report_events
from certified_cost import HealthCost, outward, real
from certified_first_packet import first_packet_bounds


HERE = Path(__file__).resolve().parent
PRIOR = "JFI_A0_direct_joint_risk_feasibility"
OFFLINE = "JFI_defensive_offline_validation"
FRONTIER = "JFI_secure_estimation_accuracy_safety_frontier"
NOMINAL = "JFI_V12_budget_and_risk_aware_compensation_20261118"
BASELINE = "e9ec9e22f04ea10c0bd3a14b11c60b31ccf1cda8"
COST_POINTS = ("anchor@0.1%","anchor@0.5%","anchor@1%","anchor@2%")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(base):
    index = read(HERE/"dependency_lock.json")
    assert index["baseline"]==BASELINE
    for name,row in index["files"].items():
        payload = (base/name).read_bytes()
        if row["normalize_CRLF"]:
            payload = payload.replace(b"\r\n",b"\n")
        if sha256(payload).hexdigest()!=row["sha256"]:
            raise ValueError(f"Frozen input differs: {name}")
    return len(index["files"])


def cost_record(cost, frozen):
    critical = real("1/10").sqrt()/2
    threshold_mse = cost.at(critical)
    zero = {"radius_interval":outward(critical),"mse_interval":outward(threshold_mse),
            "pct_above_v0_interval":outward(100*(threshold_mse/cost.v0-1))}
    assert threshold_mse > real("51/50")*cost.v0
    actual = {}
    for label in ("sat_nominal_matched",*COST_POINTS):
        radius = frozen[label]["radius"]
        level = cost.at(radius)
        actual[label] = {"radius":radius,"mse_interval":outward(level),
                         "pct_above_v0_interval":outward(100*(level/cost.v0-1))}
        assert abs(float(level)-frozen[label]["nominal_mse"])<2e-12
    return {"full_floor":outward(cost.full),"v0_mse":outward(cost.v0),
            "critical":zero,"frozen_operating_points":actual}


def reachability_check(cost, first, scale_sq):
    _, hcov, v, ky = cost.stages[7]
    size = real(scale_sq*(first.T*v.inv()*first)[0])
    state = real(scale_sq*(first.T*ky.T*ky*first)[0])
    limit = 8*real(20).log()
    assert state > 0 and size/4<limit
    assert real("1/10") < limit*state/size
    return {"minimum_base_whitened_square":outward(size/4),
            "sufficient_strict_inequality_D_ceiling":outward(limit*state/size),
            "history_gap_covariance_max_diagonal":float(max(hcov.diagonal()))}


def nominal_confidence(base, name, comparison, health, frontier, scores, registration):
    v12 = read(base/NOMINAL/"registration.json")
    old = read(base/NOMINAL/"results/nominal_audit.json")[name]
    holdout = read(base/FRONTIER/"registration.json")["holdout"]
    seeds = [v12["samples"][name]["nominal_certificate_seed"], holdout["seed"][name]]
    assert len(set(seeds))==2
    array = np.load(base/NOMINAL/f"results/{name}_bounded_health_scores.npy",allow_pickle=False)
    assert array.shape==(v12["nominal_certificate"]["paths_per_model"],4)
    past_scores = array[:,1]
    assert len(scores)==holdout["paths_per_model_and_profile"]
    floor = float(health.full)
    np.testing.assert_allclose(floor+past_scores.mean(),old["policies"]["V1a"]["average_mse_model_estimate"],atol=2e-14)
    front_mean = comparison["cases"]["healthy"]["methods"]["V1a"]["health_expectation_MSE_conditional_MC"]
    np.testing.assert_allclose(floor+scores.mean(),front_mean,atol=2e-14)
    clip = registration["health_clip"]
    confidence = health_interval(np.concatenate([past_scores,scores]),health.full,clip,
                                 outward(health.clipping_bias(clip,10))[1],
                                 registration["health_family_error"]/4)
    confidence["saved_seed_families"] = seeds
    confidence["saved_two_block_means"] = [float(floor+x.mean()) for x in (past_scores,scores)]
    confidence["equal_health_cost_radius_interval"] = matching_interval(health,frontier,confidence)
    return confidence


def plot_cost(records, models, frontier, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "e9ec9e2-health-curve"
    fig,ax = plt.subplots(figsize=(7.2,4.4),layout="constrained")
    x = np.linspace(0,0.6,201)
    for name, color in (("unstable","#21629e"),("stable","#bc561a")):
        cost = models[name]
        eig = np.array([[float(a),float(b)] for a,b in cost.eigenvalues])
        trace = [float(cost.full)]
        y = [100*(frontier.rms_cost(r,trace,eig).mean()/float(cost.v0)-1) for r in x]
        ax.plot(x,y,color=color,label=name.capitalize())
        for point in records[name]["frozen_operating_points"].values():
            ax.scatter(point["radius"],np.mean(point["pct_above_v0_interval"]),s=22,color=color,zorder=4)
    ax.axvline(np.sqrt(.1)/2,ls="--",lw=1.2,color="#414141",label="Zero-risk boundary")
    ax.axhline(2,color="#777777",ls=":",lw=1.2,label="2% budget")
    ax.set(xlabel="Saturation radius",ylabel="Mean healthy MSE increase over V0 (%)",
           xlim=(0,.6))
    ax.legend(frameon=False,ncol=2)
    ax.grid(axis="y",alpha=.22)
    fig.savefig(path,format="svg",metadata={"Date":None})
    plt.close(fig)


def evaluate(base, out, health_only=False):
    sys.dont_write_bytecode = True
    recorded_count = verify(base)
    sys.path[:0] = [str(base/PRIOR), str(base/OFFLINE)]
    last = import_module("run_joint_analysis")
    archive = import_module("replay_evaluator")
    integrals = import_module("joint_integral")
    receiver,common,evidence = archive.load_receiver_modules(base)
    frontier = import_module("frontier")
    spec = read(base/archive.V1/"registration.json")
    assert (spec["evaluation_onset"],spec["template_duration"],spec["horizon"],
            spec["danger_squared"],spec["event_false_alarm"])==(7,10,32,.1,.05)
    assert spec["watermark"]=={"lambda":.5,"q":.5}
    source = read(base/PRIOR/"results/results.json")["models"]
    comparison = read(base/FRONTIER/"results/comparison.json")["models"]
    config = read(HERE/"registration.json")
    assert config["baseline"]==BASELINE
    out.mkdir(exist_ok=True)
    models = {name:HealthCost(base/archive.MODEL,name,spec) for name in archive.MODELS}
    records = {name:cost_record(cost,source[name]["nominal_cost_numerical_evaluation"])
               for name,cost in models.items()}
    plot_cost(records,models,frontier,out/"health_cost_curve.svg")
    result = {"baseline":BASELINE,"frozen_inputs_verified_before_and_after":recorded_count,
              "added_synthetic_paths":0,"models":records}
    for name, row in records.items():
        print(name,"zero-risk critical MSE",row["critical"]["mse_interval"],
              "percent",row["critical"]["pct_above_v0_interval"],flush=True)
    if not health_only:
        for name, cost in models.items():
            previous = source[name]
            row = records[name]
            methods = [m["name"] for m in comparison[name]["method_table"]]
            scores,events,n = archive_events(base/FRONTIER/f"results/{name}_paired.npz",methods,32)
            assert tuple(events)==("V1a",*COST_POINTS)
            row["archived_risk_intervals"] = report_events(events,n,config["statistics"]["risk_family_error"]/52)
            row["nominal_matching_confidence"] = nominal_confidence(
                base,name,comparison[name],cost,frontier,scores,config["statistics"])
            settings = evidence.Settings.from_registration(spec)
            prepared = evidence.PreparedModel(common.load_model(name,base/archive.MODEL),settings)
            saved, exact, scale_sq, _ = last.recorded_template(base,archive,name,settings)
            row["parametric_reachability"] = reachability_check(cost,exact[0],scale_sq)
            radii = {label:entry["radius"] for label,entry in row["frozen_operating_points"].items()}
            frozen_integral = integrals.JointIntegral(prepared,saved,radii)
            row["original_archived_replay"] = last.cross_check(base,archive,name,receiver,prepared,spec,frozen_integral)
            repro = integrals.quadrature(frozen_integral)
            before = previous["numerical_integral"]["replications"]
            assert list(before)==list(repro["replications"])
            for size, runs in before.items():
                for a,b in zip(runs,repro["replications"][size],strict=True):
                    np.testing.assert_array_equal(a["risk_by_method_by_age"],b["risk_by_method_by_age"])
            row["frozen_sobol_indicators_reproduced"] = True
            print(name,"archived fixed-onset paths",n,"first-packet certification started",flush=True)
            row["certified_first_packet"] = first_packet_bounds(cost,7,exact[0],scale_sq,radii,config["certificate"])
            print(name,"certified bounds",row["certified_first_packet"]["bounds"],flush=True)
    verify(base)
    path = out/("health_only.json" if health_only else "results.json")
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=HERE.parent)
    parser.add_argument("--health-only",action="store_true")
    args = parser.parse_args()
    evaluate(args.baseline.resolve(),HERE/"results",args.health_only)


if __name__=="__main__":
    main()
