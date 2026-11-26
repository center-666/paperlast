"""Independent path-archive readback, paired precision, and descriptive loss tails."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def verify(folder):
    run=json.loads((folder/"RUN.json").read_text())
    tags=run["arms"]
    with (folder/"absolute.csv").open() as f:
        means={r["arm"]:r for r in csv.DictReader(f) if r["group"]=="all"}
    with (folder/"contrasts.csv").open() as f:
        contrasts={(r["comparison"],r["metric"]):r for r in csv.DictReader(f) if r["group"]=="all"}
    primary=[];early={t:[] for t in tags};loss={t:[] for t in tags};tails=[];per_key=[]
    for filename,sha in run["archives"].items():
        assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==sha
    for model in ("unstable","stable"):
        cut=run["thresholds"][model]
        with np.load(folder/f"{model}_0_0.npz") as cal,np.load(folder/f"{model}_1_0.npz") as healthy:
            for arm in tags:
                z=np.sort(cal[arm+"_max"])
                for i,a in enumerate((.025,.05,.075)):
                    assert z[int(np.ceil((len(z)+1)*(1-a)))-1]==cut[arm][i]
                if arm.endswith("_plugin"):
                    np.testing.assert_array_equal(healthy[arm+"_max"],healthy["none_plugin_max"])
                    np.testing.assert_array_equal(healthy[arm+"_loss"],healthy["none_plugin_loss"])
        for i in range(6):
            with np.load(folder/f"{model}_2_{i}.npz") as ar:
                perarm={}
                for arm in tags:
                    hit=ar[arm+"_delay"]<3
                    np.testing.assert_array_equal(hit,ar[arm+"_peak"]>cut[arm][1])
                    perarm[arm]=hit.sum(1)/6
                    early[arm].append(perarm[arm].mean())
                    a=ar[arm+"_loss"][:,2]
                    loss[arm].append(a.mean())
                    q=np.quantile(a,[.5,.95,.99,1])
                    tails.append({"model":model,"angle":(0,-45,45)[i//2],"onset":(50,65)[i%2],"arm":arm,
                                  "median":q[0],"p95":q[1],"p99":q[2],"max":q[3],"mean":a.mean(),
                                  "top_1pct_share":np.sort(a)[-int(np.ceil(.01*len(a))):].sum()/a.sum()})
                primary.append(perarm["negative_plugin"]-perarm["sign_plugin"])
    for t in tags:
        np.testing.assert_allclose(np.mean(early[t]),float(means[t]["early"]),atol=1e-12)
        np.testing.assert_allclose(np.mean(loss[t]),float(means[t]["disturbance_affected"]),atol=1e-11)
    obs=np.mean(primary);n=run["paths_per_scene"];v=np.var(primary,axis=1,ddof=1)
    r=contrasts["negative_plugin-sign_plugin","early"]
    assert abs(obs-float(r["difference"]))<1e-12
    with (folder/"loss_tails.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(tails[0]));w.writeheader();w.writerows(tails)
    return {"archives_verified":len(run["archives"]),"paths_per_scene":n,"primary_direct":obs,
            "parametric_reference_se":float(np.sqrt(v.sum()/144/n)),
            "actual_bootstrap_halfwidth":(float(r["upper"])-float(r["lower"]))/2}


if __name__=="__main__":
    report={s:verify(HERE/s) for s in ("dev","confirm")}
    print(json.dumps(report,indent=2))
    (HERE/"VERIFICATION.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
