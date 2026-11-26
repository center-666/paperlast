"""Scenario-stratified paired bootstrap and the prospective sample size lock."""
import argparse
import csv
import json
import math
import numpy as np
from scipy.stats import beta
import experiment as ex

HERE=ex.HERE


def comparisons():
    result={}
    def add(a,b):
        result[a+"-"+b]={a:1,b:-1}
    for mode in ("plugin","exact","minus","plus"):
        for a,b in (("negative","sign"),("negative","iid"),("iid","sign")):
            add(a+"_"+mode,b+"_"+mode)
    for m in ("minus","plus"):
        result["interaction_"+m]={"negative_"+m:1,"iid_"+m:-1,"negative_exact":-1,"iid_exact":1}
        for s in ("negative","iid","sign"):add(s+"_"+m,s+"_exact")
    for m in ("plugin","exact"):
        for s in ex.SCHEMES[1:]:add(s+"_"+m,"none_"+m)
        for s in ("diagonal","quarter","iid_eq","negative_eq"):add(s+"_"+m,"sign_"+m)
        for s in ("iid","negative"):add(s+"_eq_"+m,s+"_"+m)
    for s in ex.SCHEMES:add(s+"_exact",s+"_plugin")
    return result


def cp(k,n):
    return [0 if k==0 else beta.ppf(.025,k,n-k+1),1 if k==n else beta.ppf(.975,k+1,n-k)]


def csvout(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summary(split):
    path=HERE/split
    state=ex.read(path/"RUN.json")
    for file,digest in state["archives"].items():
        assert ex.old.base.digest(path/file)==digest
    sample=[];healthy=[];resources=[];sensitivity=[];scenes=[]
    rng=np.random.default_rng([20261126,17291,state["split"],991])
    n=state["paths_per_scene"]
    counts={a:[] for a in ex.TAGS}
    for mi,name in enumerate(ex.MODEL):
        th=state["thresholds"][name]
        with np.load(path/f"{name}_1_0.npz") as ar,np.load(path/f"{name}_0_0.npz") as cal:
            for s in ex.SCHEMES:
                per_path=ar[s+"_tx_energy"]/ar["raw_energy"]
                resources.append({"model":name,"scheme":s,"pooled_tx_over_raw":ar[s+"_tx_energy"].sum()/ar["raw_energy"].sum(),
                                  "mean_path_ratio":per_path.mean(),"p05":np.quantile(per_path,.05),"p95":np.quantile(per_path,.95),
                                  "health_schedule_s":ar[s+"_key_s"].sum()})
            for a in ex.TAGS:
                h=ar[a+"_max"];fwer=[(h>t).mean() for t in th[a]]
                k=int((h>th[a][1]).sum());n0=len(h);ci=cp(k,n0)
                counts[a].append(fwer)
                healthy.append({"model":name,"arm":a,"alarm_paths":k,"n":n0,"FWER":k/n0,
                                "lower":ci[0],"upper":ci[1],"health5":ar[a+"_loss"][:,0].mean(),
                                "health50":ar[a+"_loss"][:,1].mean(),"cal_score_s":cal[a+"_score_s"].sum(),
                                "holdout_score_s":ar[a+"_score_s"].sum(),"holdout_filter_s":ar[a+"_filter_s"].sum()})
        for scene in range(6):
            with np.load(path/f"{name}_2_{scene}.npz") as ar:
                methods=[]
                for a in ex.TAGS:
                    d=ar[a+"_delay"];peak=ar[a+"_peak"];loss=ar[a+"_loss"]
                    np.testing.assert_array_equal(d<3,peak>th[a][1])
                    arr=np.column_stack([(d<3).mean(1),(d.min(1)<3),ar[a+"_prior"],d.mean(1),ar[a+"_eligible"],loss])
                    methods.append(arr)
                    rows={"model":name,"angle":(0,-45,45)[scene//2],"onset":(50,65)[scene%2],"arm":a}
                    scenes.append({**rows,**dict(zip(ex.LOSSES,arr.mean(0))),"score_s":ar[a+"_score_s"].sum(),"filter_s":ar[a+"_filter_s"].sum()})
                    for k,alpha in enumerate(ex.ALPHA):
                        sensitivity.append({**rows,"target":alpha,"healthy_FWER":counts[a][mi][k],
                                            "threshold":th[a][k],"early":(peak>th[a][k]).mean()})
                sample.append(np.stack(methods,axis=1))
    cells=np.stack(sample)
    groups={"all":list(range(12)),"unstable":list(range(6)),"stable":list(range(6,12)),
            "onset50":list(range(0,12,2)),"onset65":list(range(1,12,2))}
    avgs=[];deltas=[];boot=[]
    comp=comparisons()
    # Apply each resampling matrix once across every arm and outcome of that scene.
    for x in cells:
        weights=rng.multinomial(n,np.full(n,1/n),size=10000).astype(float)/n
        boot.append((weights@x.reshape(n,-1)).reshape(10000,len(ex.TAGS),-1))
    for group,ids in groups.items():
        av=cells[ids].mean((0,1)); draws=np.mean([boot[i] for i in ids],axis=0)
        for i,a in enumerate(ex.TAGS):avgs.append({"group":group,"arm":a,**dict(zip(ex.LOSSES,av[i]))})
        for k,weights in comp.items():
            d=sum(w*draws[:,ex.TAGS.index(a)] for a,w in weights.items())
            point=sum(w*av[ex.TAGS.index(a)] for a,w in weights.items())
            ci=np.quantile(d,[.025,.975],axis=0)
            for i,metric in enumerate(ex.LOSSES):
                deltas.append({"group":group,"comparison":k,"metric":metric,"difference":point[i],"lower":ci[0,i],"upper":ci[1,i]})
    for file,rows in (("absolute",avgs),("contrasts",deltas),("health",healthy),("resources",resources),("sensitivity",sensitivity),("scenes",scenes)):
        csvout(path/(file+".csv"),rows)
    if split=="dev":
        d=cells[:,:,ex.TAGS.index("negative_plugin"),0]-cells[:,:,ex.TAGS.index("sign_plugin"),0]
        required=(1.96/.01)**2*d.var(axis=1,ddof=1).sum()/144
        lock={"n":32*math.ceil(max(512,required)/32),"calculated_n":required,"variances":d.var(axis=1,ddof=1).tolist(),
              "primary":"negative_plugin-sign_plugin","dev_hash":ex.old.base.digest(path/"RUN.json"),
              "frozen_sources":state["source_hashes"],"development_code":state["head"],"energy_scale":ex.read(HERE/"ENERGY_LOCK.json")}
        file=HERE/"CONFIRM_LOCK.json"
        if file.exists() and ex.read(file)!=lock:raise ValueError("A different confirmation lock already exists")
        ex.old.base._atomic_json(file,lock)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=("dev","confirm"));summary(p.parse_args().split)
