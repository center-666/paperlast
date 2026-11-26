"""Independent path-NPZ audit for the locked tail event."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
import experiment as exp

HERE=exp.HERE


def interval(k,n):
    return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]


def audit(split):
    name="tail_dev" if split=="dev" else "tail_confirm"
    folder=HERE/name
    manifest=json.loads((folder/"RUN.json").read_text())
    totals={"all":{"event":0,"n":0,"compatible":0,"large_after_alarm":0},
            "unstable":{"event":0,"n":0,"compatible":0,"large_after_alarm":0},
            "stable":{"event":0,"n":0,"compatible":0,"large_after_alarm":0}}
    scenes={}
    for f,digest in manifest["archives"].items():
        assert exp.old.base.digest(folder/f)==digest
    for model in exp.MODEL:
        for scene in range(6):
            filename = f"{model}_{scene}.npz"
            with np.load(folder/filename) as a:
                q=a["quarter_exact_attack_mse"];n=a["negative_exact_attack_mse"]
                e=(q>1)&(q>=10*n)
                compatible=a["quarter_state_compatible_window_count"].sum(axis=1)>0
                large=a["quarter_exact_large_after_alarm_count"]>0
                np.testing.assert_array_equal(e&compatible,e)
                np.testing.assert_array_equal(e&large,e)
                row={"tail_event":int(e.sum()),"n":len(e),
                     "exact95":interval(int(e.sum()),len(e)),
                     "tail_event_with_blind_window":int((e&compatible).sum()),
                     "tail_event_with_large_late_accept":int((e&large).sum()),
                     "quarter_mse_mean":float(q.mean()),"negative_mse_mean":float(n.mean()),
                     "quarter_latch_mse_mean":float(a["quarter_exact_latch_attack_mse"].mean()),
                     "quarter_max":float(q.max()),"negative_max":float(n.max()),
                     "archive_sha256":manifest["archives"][filename]}
                scenes[f"{model}_{scene}"]=row
                for group in ("all",model):
                    for key,val in (("event",int(e.sum())),("n",len(e)),
                                    ("compatible",int(compatible.sum())),
                                    ("large_after_alarm",int(large.sum()))):
                        totals[group][key]+=val
    out={"split":split,"run_sha256":exp.old.base.digest(folder/"RUN.json"),
         "verification_source_sha256":exp.old.base.digest(HERE/"verify_tail.py"),
         "metadata_correction":"Each scene digest now refers to its own archive. The previous report displayed the final archive digest in every scene row. The archive validation loop and event calculations were already correct.",
         "event_definition":"quarter matched attack MSE >1 and >=10x fixed-negative matched attack MSE",
         "scenes":scenes,"totals":totals}
    exp.old.base._atomic_json(folder/"VERIFY_INDEPENDENT.json",out)
    return out


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));a=p.parse_args()
    x=audit(a.split);print(json.dumps(x["totals"],indent=2))
