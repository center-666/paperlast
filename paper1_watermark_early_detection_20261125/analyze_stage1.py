"""Analyze whole-path paired records and retain the provenance of all comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
import study

NBOOT=10000
C1={"B-A":{"B":1,"A":-1},"D-C":{"D":1,"C":-1},
    "interaction":{"D":1,"C":-1,"B":-1,"A":1},"C-A":{"C":1,"A":-1},
    "D-B":{"D":1,"B":-1},"C-E":{"C":1,"E":-1},"D-F":{"D":1,"F":-1}}


def interval(k,n):
    return [beta.ppf(.025,k,n-k+1) if k else 0., beta.ppf(.975,k+1,n-k) if k<n else 1.]


def resampling(n, count, rng, reps):
    return [rng.multinomial(n,np.full(n,1/n),size=reps).astype(float)/n for _ in range(count)]


def scalar_summary(x, distribution):
    return {"difference":float(x.mean()),"paired95":np.quantile(distribution,[.025,.975]).tolist(),
            "bonferroni_lower_3":float(np.quantile(distribution,.05/3))}


def summarize(directory, reps=NBOOT):
    manifest=json.loads((directory/"RUN.json").read_text())
    for f,h in manifest["archives"].items():
        if study.base.digest(directory/f)!=h: raise ValueError(f)
    ident=manifest["identity"]; split=ident["split"]
    labels=[a[0] for a in ident["arms"]]
    cs=C1 if "A" in labels else {
        left+"-"+right:{left:1,right:-1} for kind in ("plugin","matched")
        for left,right in (("radial_"+kind,"fixed_"+kind),("radial_"+kind,"random_"+kind))
        if left in labels}
    seed=[20261125,8317,split,9001]
    rng=np.random.default_rng(seed)
    report={"split":split,"n":ident["paths_per_scene"],"seed":seed,"bootstrap":reps,
            "run_sha256":study.base.digest(directory/"RUN.json"),"health":{},"scenes":{},
            "means":{},"contrasts":{},"geometry":{},"precision_n":{},"timing_seconds":{}}
    metric_names=["early1","early2","early3","early5","early8","any_delay","eligible","prior","miss",*study.LOSSES]
    cells=[]
    for m in study.base.MODELS:
        ts=manifest["thresholds"][m]
        with np.load(directory/f"{m}_cal_0.npz") as calibration, np.load(directory/f"{m}_health_0.npz") as health:
            n=len(health["path_id"]); health_rows={};report["health"][m]=health_rows
            hw=resampling(n,1,rng,min(reps,2000))[0]
            for lab in labels:
                counts=[int((health[lab+"_max"]>t).sum()) for t in ts[lab]]
                health_rows[lab]={"alarms":counts[1],"n":n,"FWER":counts[1]/n,
                    "exact95":interval(counts[1],n),"sensitivity_FWER":[v/n for v in counts],
                    **{metric:float(health[lab+"_"+metric].mean()) for metric in study.LOSSES if metric.startswith("post20")}}
                entry=report["timing_seconds"].setdefault(lab,{"cal_score":0.,"health_score":0.,"eval_score":0.,"health_replay":0.,"eval_replay":0.})
                for key,archive,suffix in (("cal_score",calibration,"score"),("health_score",health,"score"),("health_replay",health,"replay")):
                    entry[key]+=float(archive[lab+"_"+suffix+"_cpu_wall_s"].sum())
            health_rows["paired"]={}
            for name,weights in cs.items():
                paired={}
                for metric in ("post20_affected_mse","post20_network_mse"):
                    diff=sum(w*health[lab+"_"+metric] for lab,w in weights.items())
                    paired[metric]=scalar_summary(diff,hw@diff)
                health_rows["paired"][name]=paired
        for s in range(6):
            scene=f"{m}_eval_{s}"; rows={};report["scenes"][scene]=rows
            with np.load(directory/(scene+".npz")) as ar:
                arms=[]
                for lab in labels:
                    delay=ar[lab+"_delay"]; early=ar[lab+"_early_max"]
                    np.testing.assert_array_equal(delay<3,early>ts[lab][1])
                    values=np.column_stack([*(np.mean(delay<h,axis=1) for h in (1,2,3,5,8)),
                        delay.min(1),ar[lab+"_eligible"],ar[lab+"_prior"],(delay==60).mean(1),
                        *(ar[lab+"_"+metric] for metric in study.LOSSES)])
                    arms.append(values)
                    rows[lab]={**dict(zip(metric_names,values.mean(0).tolist())),
                        "per_link_delay60":delay.mean(0).tolist(),"per_link_miss":(delay==60).mean(0).tolist(),
                        "any_link_by3":float((delay.min(1)<3).mean()),
                        "sensitivity_early3":[float((early>t).mean()) for t in ts[lab]]}
                    for which in ("score","replay"):
                        report["timing_seconds"][lab]["eval_"+which]+=float(ar[lab+"_"+which+"_cpu_wall_s"].sum())
                cells.append(np.stack(arms,axis=1))
                for scheme in ("negative_fixed","negative_radial_bin","negative_random_bin"):
                    if scheme+"_psi" in ar:
                        report["geometry"].setdefault(scene,{})[scheme]={}
                        for key in ("psi","ungated_packet_sq"):
                            x=ar[scheme+"_"+key]
                            report["geometry"][scene][scheme][key]={"mean":x.mean((0,2)).tolist(),"q10":np.quantile(x,.1,axis=(0,2)).tolist()}
    cells=np.stack(cells)
    weights=resampling(cells.shape[1],12,rng,reps)
    for group,idx in (("all",list(range(12))),("unstable",list(range(6))),("stable",list(range(6,12)))):
        report["means"][group]={a:dict(zip(metric_names,cells[idx,:,i,:].mean((0,1)).tolist())) for i,a in enumerate(labels)}
        report["contrasts"][group]={}
    for name,contrast in cs.items():
        diff=sum(w*cells[:,:,labels.index(a),:] for a,w in contrast.items())
        boot=np.stack([w@v for w,v in zip(weights,diff)])
        for group,sl in (("all",slice(None)),("unstable",slice(0,6)),("stable",slice(6,12))):
            bs=boot[sl].mean(0)
            report["contrasts"][group][name]={key:scalar_summary(diff[sl,:,i],bs[:,i]) for i,key in enumerate(metric_names)}
        early=diff[:,:,2]
        report["precision_n"][name]=int(np.ceil(1.96**2*early.var(1,ddof=1).sum()/(12**2*.01**2)))
        report["contrasts"]["all"][name]["early3"]["onset50_65"]= [float(early[i::2].mean()) for i in (0,1)]
    study.base._atomic_json(directory/"analysis.json",report)
    return report


def lock():
    dev=[json.loads((study.HERE/study.LABELS[s]/"analysis.json").read_text()) for s in (1,3)]
    eligible=[]
    for kind in ("plugin","matched"):
        x=dev[1]["contrasts"]["all"][f"radial_{kind}-fixed_{kind}"]["early3"]
        if x["difference"]>0 and min(x["onset50_65"])>0: eligible.append((x["difference"],kind))
    chosen=max(eligible,key=lambda x:x[0])[1] if eligible else None
    def size(n):return int(32*np.ceil(max(512,n)/32))
    out={"development_sha256":{str(s):study.base.digest(study.HERE/study.LABELS[s]/"analysis.json") for s in (1,3)},
         "registration_sha256":study.base.digest(study.HERE/"REGISTRATION.json"),
         "2_n":size(max(dev[0]["precision_n"][k] for k in ("interaction","B-A","D-C"))),
         "4_n":size(dev[1]["precision_n"][f"radial_{chosen}-fixed_{chosen}"]) if chosen else None,
         "stage2_receiver":chosen}
    path=study.HERE/"CONFIRM_LOCK.json"
    if path.exists() and json.loads(path.read_text())!=out: raise ValueError("A different lock already exists")
    study.base._atomic_json(path,out)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("job",choices=("1","2","3","4","lock"))
    job=parser.parse_args().job
    if job=="lock":print(lock())
    else:summarize(study.HERE/study.LABELS[int(job)])
