"""Check development action archives against registered M7 graphs and policy requests."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("D2_D1_DIR",r"C:\Users\zhongxin\shixidoc\.d1_registration_worktree\paper1_D1_link_value_20261123")
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import task3

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"class_weight_development"
METHODS=("unified","topology_two","three_class","frozen_T","saturation")


def main():
    fit=json.loads((OUT/"FIT_PARAMETERS.json").read_text(encoding="utf-8"))
    receipt=json.loads((OUT/"EVALUATION.json").read_text(encoding="utf-8"))
    archive_path=OUT/"pathwise_actions.npz"
    assert task3.digest(archive_path)==receipt["pathwise_actions_sha256"]
    evaluation={(r["config"],r["fixture"],r["method"]):r
                for r in csv.DictReader((OUT/"evaluation.csv").open(encoding="utf-8"))}
    q=np.asarray(fit["grid"],dtype=float)
    total={"attack_paths":0,"health_paths":0,"alarms":0,"M7_violations":0,"projections":0}
    with np.load(archive_path) as arrays:
        for index,config in enumerate(task3.QUALITY_ORDER):
            record=task3.read_quality_model(index)
            graph=task3.configure(index)[1]
            for fixture in ("continuous","burst","health"):
                g=arrays[f"gates_{index:02}_{fixture}"][:,:,6].astype(bool)
                alarm_slice=slice(50,110) if fixture!="health" else slice(20,150)
                total["alarms"]+=int(g[:,alarm_slice].sum())
                paths=g.shape[0]
                total["attack_paths" if fixture!="health" else "health_paths"]+=paths
                for method in METHODS:
                    actions=arrays[f"actions_{index:02}_{fixture}_{method}"].astype(int)
                    requested=arrays[f"requested_{index:02}_{fixture}_{method}"].astype(int)
                    if method=="unified":
                        request=float(fit["unified_q"])
                        request_idx=int(np.argmin(abs(q-request)))
                        expected=np.where(g,request_idx,len(q)-1)
                    elif method=="topology_two":
                        request=fit["topology_two_q"]["U" if record["topology"]=="U" else "R"]
                        request_idx=int(np.argmin(abs(q-request)))
                        expected=np.where(g,request_idx,len(q)-1)
                    elif method in ("three_class","frozen_T"):
                        if record["topology"]=="U":
                            request=fit["three_class_q"]["U_unstable" if record["dynamics"]=="unstable" else "U_stable"]
                        else:
                            request=fit["three_class_q"]["R"]
                        request_idx=int(np.argmin(abs(q-request)))
                        expected=np.where(g,request_idx,len(q)-1)
                    else:
                        raw=arrays[f"raw_weight_{index:02}_{fixture}_saturation"]
                        lower=np.searchsorted(q,raw,side="left").clip(0,len(q)-1)
                        upper=np.maximum(lower-1,0)
                        lo,hi=abs(q[lower]-raw),abs(q[upper]-raw)
                        sat_idx=np.where(lo<=hi+1e-12,lower,upper)
                        expected=np.where(g,sat_idx,len(q)-1)
                    np.testing.assert_array_equal(requested,expected)
                    hist=np.full(paths,7**6-1,dtype=np.int64)
                    violations=projections=0
                    for tick in range(1,150):
                        action=actions[:,tick]
                        legal=graph[hist]
                        bad=g[:,tick]&~legal[np.arange(paths),action]
                        violations+=int(bad.sum())
                        wanted=requested[:,tick]
                        projections+=int(np.sum(g[:,tick]&(wanted!=action)))
                        selected=legal & (np.arange(len(q))[None,:]>=wanted[:,None])
                        exp_action=selected.argmax(axis=1)
                        np.testing.assert_array_equal(action,exp_action)
                        hist=(hist*7+action)%7**6
                    row=evaluation[config,fixture,method]
                    assert violations==int(row["M7_violations"])==0
                    assert projections==int(row["M7_projections"])
                    total["M7_violations"]+=violations
                    total["projections"]+=projections
    result={"status":"PASS_pathwise_action_archive","methods":list(METHODS),
            "configurations":12,"evaluation_paths":total["attack_paths"],
            "healthy_paths":total["health_paths"],"target_alert_decisions":total["alarms"],
            "M7_violations":total["M7_violations"],"M7_projection_decisions":total["projections"],
            "action_archive_sha256":task3.digest(archive_path),
            "fit_sha256":task3.digest(OUT/"FIT_PARAMETERS.json"),
            "evaluation_sha256":task3.digest(OUT/"EVALUATION.json")}
    (OUT/"ACTION_VERIFY.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
