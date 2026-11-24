"""Path-clustered reporting and shared-feature, held-out-structure diagnostic."""
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import norm, spearmanr
from analyze_audit import Data, PAIRS, classes, ci
from audit_replay import HERE, OUT, D1, Q

def stratified(groups, baseline, alternative, period, margin=0., family=1):
    a=[g[:,alternative,period] for g in groups]
    b=[g[:,baseline,period] for g in groups]
    den=np.mean([v.mean() for v in b])
    num=np.mean([v.mean() for v in a])
    ratio=num/den
    # Independent strata, paired policies within path. Delta method for ratio of means.
    def se(terms):
        return float(np.sqrt(sum(z.var(ddof=1)/len(z) for z in terms))/len(terms))
    dif=[i-j for i,j in zip(a,b)]
    resid=[i-ratio*j for i,j in zip(a,b)]
    test=[i-(1+margin)*j for i,j in zip(a,b)]
    crit=norm.ppf(1-.05/family)
    return dict(base=float(den),alternative=float(num),relative=ratio-1,
                relative_ci95=(ratio-1+np.array([-1,1])*1.96*se(resid)/den).tolist(),
                paired_difference=float(np.mean([x.mean() for x in dif])),
                margin=margin,family=family,
                upper_margin_test=float(np.mean([z.mean() for z in test])+crit*se(test)))

def features(s,r,k,use):
    x=np.log1p(s); y=np.log1p(r)
    columns=[np.ones_like(x),x,x*x]
    if use>=1:columns.extend([y,y*y,x*y])
    if use>=2:columns.extend([k*c for c in columns.copy()])
    return np.stack(columns,-1)

def structure_diagnostic(rows):
    # Fixed small shared linear predictor per candidate; all paths in a physical
    # structure excluded together. Diagnose transfer, never use it in replay decisions.
    train={};evaluate={}
    for i,row in enumerate(rows):
        k=np.log1p(row["kappa"])
        struct=i//2
        for fix in ("continuous","burst"):
            b=Data(i,"B2",fix)
            for fold in ("fit","test"):
                sl=slice(512,1024) if fold=="fit" else slice(1024,None)
                s=b["s"][sl,:41]; r=b["run"][sl,:41];mask=r>0
                target=b["Y1"][sl,:41][mask]
                target=target-target[:,[-1]]
                y60=b["Y60"][sl,:41][mask];y60=y60-y60[:,[-1]]
                y=np.hstack([target,y60])
                for use in range(3):
                    x=features(s[mask],r[mask],k,use)
                    if fold=="fit":
                        train[i,fix,use]=(x.T@x,x.T@y)
                    else:
                        evaluate[i,fix,use]=(x.mean(0),y.mean(0))
    report=[]
    for i,row in enumerate(rows):
        for fix in ("continuous","burst"):
            common=row["conditions"]["B2_"+fix]["common_prior"]
            for use in range(3):
                pairs=[p for (scene,_,method),p in train.items() if scene//2!=i//2 and method==use]
                gram=sum(p[0] for p in pairs)
                beta=np.linalg.solve(gram+1e-6*np.eye(len(gram)),sum(p[1] for p in pairs))
                x,true=evaluate[i,fix,use];pred=x@beta
                metrics=[]
                for j,label in enumerate(("Y1","Y60")):
                    curr=pred[j*7:(j+1)*7];target=true[j*7:(j+1)*7]
                    losses=np.array(common[label]["label_to_closed_origin"]["closed"])
                    metrics.append(dict(label=label,mae=float(np.abs(curr-target).mean()),
                                        rho=float(spearmanr(curr,losses).statistic),
                                        excess=float(losses[curr.argmin()]-losses.min())))
                report.append(dict(scene=i,fixture=fix,features=["score","score_run","score_run_kappa"][use],metrics=metrics))
    return report

def main():
    rows=[json.loads((OUT/f"decisions_{i:02}.json").read_text()) for i in range(12)]
    changes={}
    for source in ("B2","D0"):
        rec=[v for row in rows for key,v in row["conditions"].items() if key.startswith(source+"_")]
        for j,label in enumerate(("Y1","Y60")):
            for i,k in PAIRS:
                ref=[v["common_prior"][label]["pairs"][f"{i}_{k}"] for v in rec]
                onR=[v["common_R_history"][f"{j}_{i}_{k}"] for v in rec]
                n=sum(x["flips"] for x in onR)
                changes[f"{source}_{label}_{i}_{k}"]=dict(alarms=sum(x["alarm"] for x in onR),
                    eligible=sum(x["eligible"] for x in onR),fixed08_flips=sum(x["flips"] for x in ref),
                    common_R_flips=n,
                    own_histories_window_flips=sum(x["own_history_full_window_flips"] for x in onR),
                    common_R_flips_mean_Y1_Y60_loss_H=(
                        (np.sum([x["sums_Y1_Y60_observed_H"] for x in onR],axis=0)/n).tolist() if n else None))
    datasets={}
    for i in range(12):
        with np.load(OUT/f"path_losses_{i:02}.npz") as z:
            datasets[i]={key:z[key] for key in z.files}
    metrics={};health=[]
    for j,label in enumerate(("Y1","Y60")):
        for i,k in PAIRS:
            for src in ("B2","D0"):
                groups=[v for r in datasets.values() for key,v in r.items() if key.startswith(src+"_")]
                if src=="D0":
                    # D0 watermarks are paired on common physical paths; merge sisters before taking SEs.
                    groups=[(datasets[s]["D0_"+fix]+datasets[s+1]["D0_"+fix])/2
                            for s in (0,6) for fix in ("continuous","burst")]
                metrics[f"{src}_{label}_{i}_{k}"]={p:stratified(groups,3*j+i,3*j+k,e,m,4)
                    for e,p,m in [(1,"window",-.01),(2,"tail",.05)]}
        for scene,r in datasets.items():
            base=r["B1_reference"]
            eps=.05-.04*(rows[scene]["phi"]-1)
            for p in range(3):
                loss=r["B1_healthy"][:,3*j+p,3]
                x=np.stack([base,loss],axis=1)[:,:,None]
                health.append(dict(scene=scene,label=label,policy=p,
                                   **stratified([x],0,1,0,eps,72)))
    ranks={}
    for lab in ("Y1","Y60"):
        for key in ("label_to_closed_origin","table_to_closed_origin","label_to_full_window"):
            groups={}
            for row in rows:
                dyn,top,_,_=row["spec"];name=dyn+"_"+top
                v=[row["conditions"]["B2_"+f]["common_prior"][lab][key] for f in ("continuous","burst")]
                groups.setdefault(name,[]).extend([x["rho"] for x in v])
            ranks[lab+"_"+key]=groups
    result=dict(contrast_ids={"0":"R","1":"RH","2":"RHV"},decision_changes=changes,closed=metrics,
                health=health,rank_correlations=ranks,leave_structure_out=structure_diagnostic(rows))
    (OUT/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(changes,indent=1))
    for key,m in metrics.items():
        print(key, *[round(m["window"][n],5) for n in ("base","alternative","relative","upper_margin_test")],
              "tail_rel",round(m["tail"]["relative"],5))
    print("health worst margin upper",max(x["upper_margin_test"] for x in health))
    print(ranks)


if __name__=="__main__":
    main()
