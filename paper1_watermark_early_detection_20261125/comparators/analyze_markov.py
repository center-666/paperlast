"""Path-stratified W5 analysis and pre-locked confirmation decision."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta
import rotation_markov as study
import experiment as exp

HERE=study.HERE
BOOT=10000
SEED=[20261127,52387,5001]
ARMS=study.ARMS
SCHEMES=study.SCHEMES


def cp(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def paired_boot(cell_values,rng):
    # One vector per fixed scenario. Each draw resamples whole paths within scenario.
    n=len(cell_values[0])
    draws=[]
    for values in cell_values:
        index=rng.integers(0,n,size=(BOOT,n))
        draws.append(values[index].mean(axis=1))
    return np.mean(draws,axis=0)


def summarize(split):
    folder=HERE/("markov_dev" if split=="dev" else "markov_confirm")
    run=json.loads((folder/"RUN.json").read_text())
    for f,h in run["archives"].items():
        if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Markov archive SHA mismatch: {f}")
    rng=np.random.default_rng([*SEED,5 if split=="dev" else 6])
    scenarios=[]
    for mi,model in enumerate(exp.MODEL):
        thresholds=run["thresholds"][model]
        cal=np.load(folder/f"{model}_calibration.npz")
        for arm in ARMS:
            maxima=np.sort(cal["max_"+arm])
            rec=thresholds[arm]
            rank=int(np.ceil((len(maxima)+1)*.95))
            if rec["one_based_rank"]!=rank or rec["threshold"]!=maxima[rank-1]:
                raise AssertionError(f"Order-statistic mismatch {model} {arm}")
        cal.close()
        health=np.load(folder/f"{model}_health_0.npz")
        health_rows={}
        for arm in ARMS:
            k=int(health[arm+"_fwer"].sum());n=len(health[arm+"_fwer"])
            health_rows[arm]={"alarms":k,"n":n,"FWER":k/n,"CP95":cp(k,n),
                              "post20_affected_mse":float(health[arm+"_post20_affected_mse"].mean()),
                              "post20_network_mse":float(health[arm+"_post20_network_mse"].mean())}
        health.close()
        for scene in range(6):
            with np.load(folder/f"{model}_attack_{scene}.npz") as data:
                row={"model":model,"scene":scene,"health":health_rows,"arms":{}}
                for arm in ARMS:
                    early=data[arm+"_early_link"].astype(bool)
                    hit=early.mean(axis=1)
                    mse=data[arm+"_attack_mse"]
                    negative=data["negative_exact_attack_mse"]
                    event=(mse>1.)&(mse>=10*negative)
                    scheme=arm.rsplit("_",1)[0]
                    row["arms"][arm]={
                        "path_early3_per_link":hit,
                        "mean_early3":float(hit.mean()),
                        "attack_mse":mse,
                        "attack_mse_mean":float(mse.mean()),
                        "recovery_mse_mean":float(data[arm+"_recovery_mse"].mean()),
                        "tail_event":event,
                        "tail_count":int(event.sum()),
                        "compatible_path":data[scheme+"_state_compatible_windows"].sum(axis=1)>0,
                        "late_large_accept":data[arm+"_large_late_accept_count"]>0,
                        "min_psi_relative":data[scheme+"_min_relative_psi"].min(axis=1),
                    }
                row["transition_count"]=data["transition_count"].sum(axis=0)
                row["state_count_iid"]=data["iid_state_count"].sum(axis=0)
                row["state_count_no_repeat"]=data["no_repeat_state_count"].sum(axis=0)
                scenarios.append(row)
    metrics=("early3","attack_mse","tail_probability","state_compatible_path_probability",
             "late_large_accepted_path_probability","min_relative_psi")
    means={};contrasts={}
    groups={"all":scenarios,
            "unstable":[s for s in scenarios if s["model"]=="unstable"],
            "stable":[s for s in scenarios if s["model"]=="stable"]}
    for group,selected in groups.items():
        means[group]={}
        for arm in ARMS:
            absolute={}
            for metric in metrics:
                if metric=="early3":
                    values=[s["arms"][arm]["path_early3_per_link"] for s in selected]
                    point=np.mean([v.mean() for v in values])
                    bs=paired_boot(values,rng)
                elif metric=="attack_mse":
                    values=[s["arms"][arm]["attack_mse"] for s in selected]
                    point=np.mean([v.mean() for v in values]);bs=paired_boot(values,rng)
                elif metric=="tail_probability":
                    values=[s["arms"][arm]["tail_event"].astype(float) for s in selected]
                    point=np.mean([v.mean() for v in values]);bs=paired_boot(values,rng)
                elif metric=="state_compatible_path_probability":
                    values=[s["arms"][arm]["compatible_path"].astype(float) for s in selected]
                    point=np.mean([v.mean() for v in values]);bs=paired_boot(values,rng)
                elif metric=="late_large_accepted_path_probability":
                    values=[s["arms"][arm]["late_large_accept"].astype(float) for s in selected]
                    point=np.mean([v.mean() for v in values]);bs=paired_boot(values,rng)
                else:
                    values=[s["arms"][arm]["min_psi_relative"] for s in selected]
                    point=np.mean([v.mean() for v in values]);bs=paired_boot(values,rng)
                absolute[metric]={"mean":float(point),"bootstrap95":np.quantile(bs,[.025,.975]).tolist()}
            means[group][arm]=absolute
        pair={}
        for left,right,label in (
            ("quarter_no_repeat_exact","quarter_iid_exact","no_repeat_minus_iid_exact"),
            ("quarter_no_repeat_plugin","quarter_iid_plugin","no_repeat_minus_iid_plugin"),
            ("quarter_no_repeat_exact","negative_exact","no_repeat_minus_negative_exact"),
            ("quarter_iid_exact","negative_exact","iid_minus_negative_exact"),
        ):
            pair[label]={}
            for metric in metrics:
                lv=[];rv=[]
                for s in selected:
                    L=s["arms"][left];R=s["arms"][right]
                    if metric=="early3":
                        lv.append(L["path_early3_per_link"]);rv.append(R["path_early3_per_link"])
                    elif metric=="attack_mse":
                        lv.append(L["attack_mse"]);rv.append(R["attack_mse"])
                    elif metric=="tail_probability":
                        lv.append(L["tail_event"].astype(float));rv.append(R["tail_event"].astype(float))
                    elif metric=="state_compatible_path_probability":
                        lv.append(L["compatible_path"].astype(float));rv.append(R["compatible_path"].astype(float))
                    elif metric=="late_large_accepted_path_probability":
                        lv.append(L["late_large_accept"].astype(float));rv.append(R["late_large_accept"].astype(float))
                    else:
                        lv.append(L["min_psi_relative"]);rv.append(R["min_psi_relative"])
                d=[a-b for a,b in zip(lv,rv)]
                point=float(np.mean([v.mean() for v in d]))
                bs=paired_boot(d,rng)
                pair[label][metric]={"difference":point,"bootstrap95":np.quantile(bs,[.025,.975]).tolist()}
        contrasts[group]=pair
    transition=np.sum([s["transition_count"] for s in scenarios],axis=0)
    state_iid=np.sum([s["state_count_iid"] for s in scenarios],axis=0)
    state_nr=np.sum([s["state_count_no_repeat"] for s in scenarios],axis=0)
    report={"split":split,"run_sha256":exp.old.base.digest(folder/"RUN.json"),
            "bootstrap_replicates":BOOT,"seed":SEED,"groups":means,"paired_contrasts":contrasts,
            "health_by_model":{m:scenarios[6*i]["health"] for i,m in enumerate(exp.MODEL)},
            "candidate":{"conditional_entropy_bits":math.log2(3),
                         "iid_entropy_bits":2.,
                         "transition_matrix":(transition/transition.sum(axis=1,keepdims=True)).tolist(),
                         "transition_counts":transition.tolist(),
                         "single_time_state_frequencies_iid":(state_iid/state_iid.sum()).tolist(),
                         "single_time_state_frequencies_no_repeat":(state_nr/state_nr.sum()).tolist(),
                         "marginal_uniform_theory":[.25]*4},
            "scenarios":[{"model":s["model"],"scene":s["scene"],
                          "arms":{a:{k:(v if np.isscalar(v) else v.tolist())
                                     for k,v in s["arms"][a].items()
                                     if k in ("mean_early3","attack_mse_mean","recovery_mse_mean","tail_count")}
                                  for a in ARMS}}
                         for s in scenarios]}
    exp.old.base._atomic_json(folder/"ANALYSIS.json",report)
    if split=="dev":
        gate=report["paired_contrasts"]["all"]["no_repeat_minus_iid_exact"]["early3"]["bootstrap95"][0]>-.01
        tail=(sum(s["arms"]["quarter_no_repeat_exact"]["tail_count"] for s in scenarios)
              <sum(s["arms"]["quarter_iid_exact"]["tail_count"] for s in scenarios))
        lock={"development_manifest_sha256":exp.old.base.digest(folder/"RUN.json"),
              "development_analysis_sha256":exp.old.base.digest(folder/"ANALYSIS.json"),
              "source_sha256":run["identity"]["source_sha256"],
              "threshold_manifest_sha256":run["identity"]["W1_threshold_sha256"],
              "noninferiority_lower95":report["paired_contrasts"]["all"]["no_repeat_minus_iid_exact"]["early3"]["bootstrap95"][0],
              "tail_event_counts":{"no_repeat":sum(s["arms"]["quarter_no_repeat_exact"]["tail_count"] for s in scenarios),
                                   "iid":sum(s["arms"]["quarter_iid_exact"]["tail_count"] for s in scenarios)},
              "development_gate":{"detection_noninferior_1pp":bool(gate),"fewer_tail_events":bool(tail)},
              "proceed":bool(gate and tail),"confirm_n_per_scene":512,
              "calibration_and_holdout_per_model":4096}
        exp.old.base._atomic_json(HERE/"MARKOV_LOCK.json",lock)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    print(json.dumps(summarize(a.split)["paired_contrasts"]["all"],indent=2))
