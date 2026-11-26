"""Path-stratified summaries for the targeted outlier recurrence study."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
import tail_study as study
import experiment as exp

HERE=study.HERE
BOOT=10000
SEED=[20261127,52387,2001]
SCHEMES=study.SCHEMES


def cp(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def summarize(split):
    folder=HERE/("tail_dev" if split=="dev" else "tail_confirm")
    manifest=json.loads((folder/"RUN.json").read_text())
    for f,h in manifest["archives"].items():
        if exp.old.base.digest(folder/f)!=h:raise ValueError(f"Archive SHA mismatch: {f}")
    cells=[];records=[]
    for mi,model in enumerate(exp.MODEL):
        for scene in range(6):
            with np.load(folder/f"{model}_{scene}.npz") as a:
                q=a["quarter_exact_attack_mse"];neg=a["negative_exact_attack_mse"]
                event=(q>1.)&(q>=10.*neg)
                for i in range(len(q)):
                    row={"model":model,"angle":(0,-45,45)[scene//2],"onset":(50,65)[scene%2],
                         "path_id":i,"tail_event":int(event[i]),
                         "quarter_attack_mse":float(q[i]),"negative_attack_mse":float(neg[i]),
                         "quarter_latch_attack_mse":float(a["quarter_exact_latch_attack_mse"][i]),
                         "quarter_min_residual_ratio":float(a["quarter_min_relative_residual_energy"][i].min()),
                         "quarter_min_residual_edge":int(a["quarter_min_relative_residual_energy"][i].argmin()),
                         "quarter_state_compatible_windows":int(a["quarter_state_compatible_window_count"][i].sum()),
                         "quarter_late_large_accepts":int(a["quarter_exact_large_after_alarm_count"][i]),
                         "quarter_accepted_last_attack_report":int(a["quarter_exact_accepted_last_report_count"][i])}
                    for s in SCHEMES:
                        row[s+"_mse"]=float(a[s+"_exact_attack_mse"][i])
                        row[s+"_delay_avg"]=float(a[s+"_exact_delay60"][i].mean())
                        row[s+"_reaccept_after_alarm"]=int(a[s+"_exact_accepted_after_alarm_count"][i])
                        row[s+"_large_after_alarm"]=int(a[s+"_exact_large_after_alarm_count"][i])
                        row[s+"_blind_windows"]=int(a[s+"_state_compatible_window_count"][i].sum())
                        row[s+"_min_relative_psi"]=float(a[s+"_min_relative_residual_energy"][i].min())
                    records.append(row)
                cells.append({
                    "model":model,"scene":scene,"quarter":q,"negative":neg,"event":event,
                    "quarter_tail":q>1,"late_reaccept":a["quarter_exact_accepted_last_report_count"]>0,
                    "quarter_latch":a["quarter_exact_latch_attack_mse"],
                    "mse":{s:a[s+"_exact_attack_mse"] for s in SCHEMES},
                    "blind":{s:(a[s+"_state_compatible_window_count"].sum(axis=1)>0) for s in SCHEMES},
                    "large_accept":{s:a[s+"_exact_large_after_alarm_count"]>0 for s in SCHEMES},
                })
    report={"status":"independent_path_tail_event_"+split,"split":split,
            "seed_namespace":manifest["identity"]["namespace"],"paths_per_cell":manifest["identity"]["paths_per_cell"],
            "run_sha256":exp.old.base.digest(folder/"RUN.json"),"event_definition":"quarter exact-key MSE>1 and >=10*negative exact-key MSE",
            "threshold_manifest_sha256":manifest["identity"]["threshold_manifest_sha256"],
            "overall":{},"models":{},"scenes":{},"paired":{},"bootstrap_seed":SEED,"bootstrap_replicates":BOOT}
    rng=np.random.default_rng(SEED)
    for group, selected in (("all",cells),("unstable",[x for x in cells if x["model"]=="unstable"]),
                            ("stable",[x for x in cells if x["model"]=="stable"])):
        q=np.concatenate([c["quarter"] for c in selected]);n=np.concatenate([c["negative"] for c in selected])
        event=(q>1)&(q>=10*n);k=int(event.sum());N=len(event)
        event_boot=[]
        for c in selected:
            local=(c["quarter"]>1)&(c["quarter"]>=10*c["negative"])
            idx=rng.integers(0,len(local),size=(BOOT,len(local)))
            event_boot.append(local[idx].mean(axis=1))
        event_ci=np.quantile(np.mean(event_boot,axis=0),[.025,.975])
        report["overall"][group]={"event_count":k,"n":N,"probability":k/N,
            "pooled_binomial_reference95":cp(k,N),
            "stratified_path_bootstrap95":event_ci.tolist(),
            "quarter_mse":{"mean":float(q.mean()),"median":float(np.median(q)),"p95":float(np.quantile(q,.95)),
                           "p99":float(np.quantile(q,.99)),"max":float(q.max())},
            "negative_mse":{"mean":float(n.mean()),"median":float(np.median(n)),"p95":float(np.quantile(n,.95)),
                            "p99":float(np.quantile(n,.99)),"max":float(n.max())},
            "quarter_gt1_count":int((q>1).sum()),
            "quarter_paths_with_large_post_alarm_accept":int(sum(c["large_accept"]["quarter"].sum() for c in selected)),
            "quarter_paths_with_last_attack_report_reaccept":int(sum(c["late_reaccept"].sum() for c in selected))}
        for s in SCHEMES:
            z=np.concatenate([c["mse"][s] for c in selected])
            report["overall"][group].setdefault("scheme_tail",{})[s]={
                "mean":float(z.mean()),"median":float(np.median(z)),"p95":float(np.quantile(z,.95)),
                "p99":float(np.quantile(z,.99)),"max":float(z.max()),
                "mse_gt1":int((z>1).sum()),
                "state_compatible_path_count":int(sum(c["blind"][s].sum() for c in selected)),
                "post_alarm_large_accept_path_count":int(sum(c["large_accept"][s].sum() for c in selected))}
    for mi,model in enumerate(exp.MODEL):
        chosen=[c for c in cells if c["model"]==model]
        q=np.concatenate([c["quarter"] for c in chosen]);n=np.concatenate([c["negative"] for c in chosen])
        event=(q>1)&(q>=10*n);k=int(event.sum());N=len(event)
        report["models"][model]={"tail_event_count":k,"n":N,"probability":k/N,"exact95":cp(k,N),
            "quarter_mse_mean":float(q.mean()),"negative_mse_mean":float(n.mean()),
            "quarter_latch_mse_mean":float(np.concatenate([c["quarter_latch"] for c in chosen]).mean())}
        for c in chosen:
            q=c["quarter"];n=c["negative"];e=(q>1)&(q>=10*n)
            report["scenes"][f"{model}_{c['scene']}"]={"event_count":int(e.sum()),"n":len(e),
                "probability":float(e.mean()),"exact95":cp(int(e.sum()),len(e)),
                "quarter_mean":float(q.mean()),"negative_mean":float(n.mean()),
                "max_quarter":float(q.max()),"max_negative":float(n.max()),
                "last_report_accepted_after_alarm":int(c["late_reaccept"].sum()),
                "latch_mean":float(c["quarter_latch"].mean())}
    # Reuse a separate deterministic stream for paired continuous outcomes.
    rng=np.random.default_rng([*SEED,1])
    for name in ("quarter","diagonal","sign","iid"):
        diffs=[];blind=[];large=[]
        for c in cells:
            diffs.append(c["mse"][name]-c["mse"]["negative"])
            blind.append(c["blind"][name].astype(float)-c["blind"]["negative"].astype(float))
            large.append(c["large_accept"][name].astype(float)-c["large_accept"]["negative"].astype(float))
        point=float(np.mean(np.concatenate(diffs)))
        boot=[]
        for di in diffs:
            indices=rng.integers(0,len(di),size=(BOOT,len(di)))
            boot.append(di[indices].mean(axis=1))
        bs=np.mean(boot,axis=0)
        report["paired"][name+"_minus_negative_mse"]={"difference":point,
            "stratified_path_bootstrap95":np.quantile(bs,[.025,.975]).tolist()}
        report["paired"][name+"_minus_negative_blind_window_path_probability"]={
            "difference":float(np.mean(np.concatenate(blind)))}
        report["paired"][name+"_minus_negative_large_accept_path_probability"]={
            "difference":float(np.mean(np.concatenate(large)))}
    exp.old.base._atomic_json(folder/"analysis.json",report)
    exp.old.base._atomic_npz(folder/"path_readback.npz",{
        "path_id":np.arange(len(records)),"rows_json":np.asarray(
            [json.dumps(row,separators=(",",":")) for row in records],dtype="U")})
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    print(json.dumps(summarize(a.split)["overall"],indent=2))
