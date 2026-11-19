"""Read-only timing audit of frozen defender output arrays."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent/"JFI_secure_estimation_accuracy_safety_frontier"
ENERGY = HERE.parent/"JFI_coded_energy_constrained_secure_estimation"
BASE = "bebc8ae84d511100a61a72725c8a4b56c42194d2"
OUTPUT = HERE/"results"
MODELS = ("unstable","stable")
METHODS = ("V0","V1a","V1.2a","V1.4","anchor@2%")
PROFILES = ("unknown_start","A1_H_replay","A1_C_replay")
HORIZON = 32
DANGER = .1
DURATION = 10


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_frozen():
    sets = {}
    for prefix,folder in (("frontier",FRONTIER),("coded_energy",ENERGY)):
        files = load(folder/"results"/"delivery_checks.json")["results_sha256"]
        for name,value in files.items():
            assert digest(folder/"results"/name)==value,(prefix,name)
        sets[prefix] = files
    return sets


def event_partition(d,a,start,stop):
    """Times use HORIZON as a right censoring code."""
    observed = d<HORIZON
    ad = a<HORIZON
    pairs = observed & ad
    assert np.all(d>=start)
    pre = observed & (d<a)
    same = pairs & (d==a)
    post = observed & (d>a)
    no_danger = ~observed
    first = observed & (d==start)
    assert np.all(pre.astype(int)+same+post+no_danger==1)
    lags = a[pairs]-d[pairs]
    values,counts = np.unique(lags,return_counts=True)
    return {
        "sample_size":int(len(d)),
        "any_danger_within_horizon":int(observed.sum()),
        "strictly_before_alarm":int(pre.sum()),
        "same_tick_as_alarm":int(same.sum()),
        "strictly_after_alarm":int(post.sum()),
        "no_danger_within_horizon":int(no_danger.sum()),
        "strict_pre_fraction_among_danger":float(pre.sum()/observed.sum()) if observed.any() else None,
        "first_packet_any_danger":int(first.sum()),
        "first_packet_strictly_before_alarm":int((first&pre).sum()),
        "first_packet_same_tick_as_alarm":int((first&same).sum()),
        "first_packet_strictly_after_alarm":int((first&post).sum()),
        "first_danger_from_start_to_stop_exclusive":int((observed&(d<stop)).sum()),
        "first_danger_at_or_after_stop":int((observed&(d>=stop)).sum()),
        "alarm_strictly_before_onset":int((ad&(a<start)).sum()),
        "alarm_on_onset":int((ad&(a==start)).sum()),
        "danger_observed_alarm_censored":int((observed&~ad).sum()),
        "alarm_observed_danger_censored":int((ad&~observed).sum()),
        "both_censored":int((~observed&~ad).sum()),
        "both_observed":int(pairs.sum()),
        "alarm_minus_danger_median_if_both_observed":float(np.median(lags)) if len(lags) else None,
        "alarm_minus_danger_histogram_if_both_observed":{
            str(int(x)):int(n) for x,n in zip(values,counts,strict=True)
        }
    }


def check_inputs(data,sample):
    a,d = (data(f) for f in ("first_alarm","first_danger"))
    t,end = (data(f) for f in ("onset","stop"))
    assert a.ndim==1 and d.ndim==2 and d.shape[0]==len(a)==len(t)==len(end)
    assert all(np.issubdtype(x.dtype,np.integer) for x in (a,d,t,end))
    assert np.all((0<=a)&(a<=HORIZON)) and np.all((0<=d)&(d<=HORIZON))
    assert np.all((0<=t)&(t<=end)&(end<=HORIZON))
    assert np.all(d>=t[:,None])
    np.testing.assert_array_equal(data("pre_alarm"),d<a[:,None])
    np.testing.assert_array_equal(data("peak_paired_squared")>=DANGER,d<HORIZON)
    np.testing.assert_array_equal(data("first_packet"),(d==t[:,None])&(d<HORIZON))
    np.testing.assert_array_equal(data("first_packet_pre_alarm"),data("first_packet")&(d<a[:,None]))
    assert sample["common_BF_alarm_count"]==int((a<HORIZON).sum())


def audit_health(data,columns,report,nominal_v0):
    h = data("health_mean_mse").mean(axis=0)
    cond = data("conditional_health_squared").mean(axis=0)
    j0 = columns["V0"]
    rows = {}
    for name,j in columns.items():
        old = report["methods"][name]
        np.testing.assert_allclose(h[j],old["health_actual_mse"][0],atol=3e-13)
        pct = 100*(h[j]/h[j0]-1)
        via_cond = 100*(cond[j]-cond[j0])/nominal_v0
        np.testing.assert_allclose(pct,old["health_actual_pct"],atol=2e-10)
        np.testing.assert_allclose(via_cond,old["health_expectation_pct_conditional_MC95"][0],atol=2e-10)
        rows[name] = {"sample_healthy_MSE":float(h[j]),"sample_increment_pct":float(pct),
                      "nominal_increment_conditional_estimate_pct":float(via_cond),
                      "nominal_mse_analytic_if_available":old.get("health_expected_MSE_exact_formula"),
                      "nominal_expected_increment_exact_formula_pct":old.get("health_expected_pct_exact_formula"),
                      "nominal_increment_upper_pct":old.get("analytic_nominal_health_penalty_upper_pct")}
    return rows


def timing_rows(data,columns,chosen,summary,full_profile):
    a = data("first_alarm")[chosen]
    d = data("first_danger")[chosen]
    t = data("onset")[chosen]
    stop = data("stop")[chosen]
    out = {}
    for name,j in columns.items():
        row = event_partition(d[:,j],a,t,stop)
        row["active_sample_mse"] = float(data("active_mean_mse")[chosen,j].mean())
        row["tail_sample_mse"] = float(data("tail_mean_mse")[chosen,j].mean())
        row["max_paired_squared_displacement_in_horizon"] = float(data("peak_paired_squared")[chosen,j].max())
        pre = d[:,j]<a
        base = d[:,columns["V1a"]]<a
        row["paired_pre_alarm_vs_v1a"] = {
            "only_this_method":int((pre&~base).sum()),
            "only_reference_v1a":int((~pre&base).sum()),
            "risk_difference_pp":100*float(pre.mean()-base.mean())
        }
        if full_profile:
            row["ever_crosses_threshold_in_tail"] = summary["methods"][name]["tail_cross_count"]
            assert row["strictly_before_alarm"]==summary["methods"][name]["pre_alarm_count"]
            assert row["first_packet_any_danger"]==summary["methods"][name]["first_packet_count"]
        out[name] = row
    return out


def audit():
    hashes = verify_frozen()
    source = FRONTIER/"results"
    registration = load(FRONTIER/"registration.json")
    assert registration["horizon"]==HORIZON and registration["D_squared"]==DANGER
    assert registration["coded_energy_limit"]==10 and registration["BF_event_budget"]==.05
    saved = load(source/"comparison.json")
    certificate = load(source/"frontier.json")
    result = {"baseline":BASE,"scope":"Receiver outcome archives only.",
              "new_trajectories":0,"live_attacker_runs":0,"offensive_parameter_searches":0,
              "horizon":HORIZON,"danger_squared":DANGER,"nominal_bf_event_alpha":.05,
              "verified_previous_results_sha256":hashes,"models":{}}
    for name in MODELS:
        model = saved["models"][name]
        names = [row["name"] for row in model["method_table"]]
        columns = {p:names.index(p) for p in METHODS}
        cases = {}
        with np.load(source/(name+"_paired.npz"),allow_pickle=False) as arr:
            def health(field):
                return arr["healthy__"+field]
            check_inputs(health,model["cases"]["healthy"])
            health_rows = audit_health(health,columns,model["cases"]["healthy"],
                                       certificate["models"][name]["v0_mse"])
            for profile in PROFILES:
                def data(field):
                    return arr[profile+"__"+field]
                t = data("onset")
                s = model["cases"][profile]
                check_inputs(data,s)
                assert len(t)==registration["holdout"]["paths_per_model_and_profile"]
                assert np.all(data("stop")-t==DURATION)
                assert set(t.tolist())==({3,7,11} if profile=="unknown_start" else {7})
                np.testing.assert_array_equal(data("health_mean_mse"),health("health_mean_mse"))
                strata = {"all_frozen_onsets":timing_rows(data,columns,np.ones(len(t),dtype=bool),s,True)}
                for k in sorted(set(t.tolist())):
                    select = t==k
                    strata[f"onset_{k}"] = timing_rows(data,columns,select,s,bool(select.all()))
                cases[profile] = strata
            old_length = arr["timing_mismatch__stop"]-arr["timing_mismatch__onset"]
            assert np.all(old_length==8)
            result["models"][name] = {"profiles":cases,"health":health_rows,
                "archived_healthy_paths":int(len(health("first_alarm"))),
                "archived_healthy_bf_alarms":int((health("first_alarm")<HORIZON).sum()),
                "excluded_eight_step_records":int(len(old_length))}
    return result


if __name__=="__main__":
    record = audit()
    save(OUTPUT/"archived_timing.json",record)
    for name,model in record["models"].items():
        for p,info in model["profiles"].items():
            print(name,p,{d:(c["any_danger_within_horizon"],c["strictly_before_alarm"],
                             c["same_tick_as_alarm"],c["strictly_after_alarm"])
                 for d,c in info["onset_7"].items()},flush=True)
