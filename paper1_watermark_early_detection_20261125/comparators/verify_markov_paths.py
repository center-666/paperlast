"""Independent audit of W5 path archives, calibration ranks and transition counts."""
import json
from pathlib import Path
import numpy as np
import rotation_markov as study
import experiment as exp


def verify(split):
    folder=study.HERE/("markov_dev" if split=="dev" else "markov_confirm")
    run=json.loads((folder/"RUN.json").read_text())
    analysis=json.loads((folder/"ANALYSIS.json").read_text())
    for f,digest in run["archives"].items():
        if exp.old.base.digest(folder/f)!=digest:raise ValueError(f"SHA mismatch: {f}")
    counts={"quarter_iid":0,"quarter_no_repeat":0,"negative":0}
    transitions=np.zeros((4,4),dtype=np.int64)
    states=np.zeros(4,dtype=np.int64)
    exact_counts={}
    for model in exp.MODEL:
        ncal=run["identity"]["calibration_per_model"]
        with np.load(folder/f"{model}_calibration.npz") as cal:
            cal_probe=cal["physical_probe"].copy()
            cal_key=cal["key_probe"].copy()
            for arm in study.ARMS:
                x=np.sort(cal["max_"+arm])
                rank=int(np.ceil((ncal+1)*.95))
                record=run["thresholds"][model][arm]
                if record["one_based_rank"]!=rank or record["threshold"]!=x[rank-1]:
                    raise AssertionError((model,arm,"calibration order statistic"))
        for scene in range(6):
            file=folder/f"{model}_attack_{scene}.npz"
            with np.load(file) as a:
                n=len(a["path_id"])
                np.testing.assert_array_equal(a["path_id"],np.arange(n))
                t=a["transition_count"].reshape(-1,4,4).sum(axis=0)
                if np.diag(t).any():raise AssertionError("No-repeat transition matrix has a diagonal count.")
                transitions+=t
                states+=a["no_repeat_state_count"].reshape(-1,4).sum(axis=0)
                negative=a["negative_exact_attack_mse"]
                for arm in ("quarter_iid_exact","quarter_no_repeat_exact","negative_exact"):
                    if arm=="quarter_iid_exact":scheme="quarter_iid"
                    elif arm=="quarter_no_repeat_exact":scheme="quarter_no_repeat"
                    else:scheme="negative"
                    event=(a[arm+"_attack_mse"]>1)&(a[arm+"_attack_mse"]>=10*negative)
                    counts[scheme]+=int(event.sum())
        with np.load(folder/f"{model}_health_0.npz") as h:
            if np.array_equal(cal_probe,h["physical_probe"]) or np.array_equal(cal_key,h["key_probe"]):
                raise AssertionError("Healthy calibration and holdout reused the same stream.")
            with np.load(folder/f"{model}_attack_0.npz") as attacked:
                if np.array_equal(cal_probe[:len(attacked["path_id"])],attacked["physical_probe"]):
                    raise AssertionError("Attack evaluation reused healthy calibration physical paths.")
            for arm in study.ARMS:
                k=int(h[arm+"_fwer"].sum())
                n=len(h[arm+"_fwer"])
                row=analysis["health_by_model"][model][arm]
                if k!=row["alarms"] or n!=row["n"]:
                    raise AssertionError((model,arm,"held-out FWER count"))
    empirical=transitions/transitions.sum(axis=1,keepdims=True)
    target=np.full((4,4),1/3);np.fill_diagonal(target,0.)
    if np.max(np.abs(empirical-target))>.003:raise AssertionError("Empirical transition law differs from no-repeat registration.")
    state_frequency=states/states.sum()
    if np.max(np.abs(state_frequency-.25))>.003:raise AssertionError("Empirical state marginal differs from uniform.")
    report={"split":split,"run_sha256":exp.old.base.digest(folder/"RUN.json"),
            "archives_verified":len(run["archives"]),"tail_event_counts":counts,
            "no_repeat_empirical_transition":empirical.tolist(),
            "no_repeat_diagonal_transitions":int(np.trace(transitions)),
            "no_repeat_state_frequency":state_frequency.tolist(),
            "conditional_entropy_bits":run["identity"]["conditional_entropy_bits"]}
    exp.old.base._atomic_json(folder/"VERIFY_INDEPENDENT.json",report)
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    print(json.dumps(verify(a.split),indent=2))
