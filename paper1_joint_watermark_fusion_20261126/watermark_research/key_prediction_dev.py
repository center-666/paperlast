"""Cross-fitted one-step disturbance predictability diagnostic on namespace-85 paths."""
import csv,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/"watermark_research"
ARCH=Path(r"D:\paperlast_task4_archive_20261127\task4_confirm")
os.environ.setdefault("D2_D1_DIR",r"C:\Users\zhongxin\shixidoc\.d1_registration_worktree\paper1_D1_link_value_20261123")
sys.path.insert(0,str(HERE))
import task3
import models_quality as quality

CONFIG="U_unstable_R35x2";INDEX=2;L=8;START=50;STOP=110
ALPHAS=(.001,.01,.1,1.,10.,100.);BOOT=20000;SEED=[20261128,9521,79]
METHODS=("zero","history_only","key_history","key_mean_prior")


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def replay_T(archive,prefix,fixture,model):
    tag="no_repeat_"
    y=archive["y"];x=archive["x"];delta=archive[tag+"delta"]
    keys=archive[tag+"keys"][:,:,6];gates=archive[tag+"gates"].astype(bool)
    actions=archive[tag+"S_fixed_table_T_actions"];n=len(x)
    states=np.zeros((n,150,4));innov=np.zeros((n,150,2));raw=(y.swapaxes(-1,-2)@model.adjacency.T).swapaxes(-1,-2)
    sender=int(model.links[6,1]);lookup={tuple(map(int,e)):i for i,e in enumerate(model.links[:6])}
    for t in range(1,150):
        prior=states[:,t-1]@model.A.T
        packet=y[:,t,sender]+delta[:,t,6]
        innov[:,t]=packet-prior@model.H[sender].T
        received=raw[:,t,26].copy()
        for (receiver,src),edge in lookup.items():
            if receiver==26:received-=gates[:,t,edge,None]*y[:,t,src]
        alarm=gates[:,t,6];q=np.where(alarm,quality.frozen.Q[actions[:,t]],1.)
        packet26=received+delta[:,t,6]+(q-1.)[:,None]*(y[:,t,sender]+delta[:,t,6])
        mode=alarm.astype(int);idx=actions[:,t]
        f=model.target["F"][mode,idx];k=model.target["K"][mode,idx];g=model.target["G"][mode,idx]
        states[:,t]=(np.einsum("bij,bj->bi",f,states[:,t-1])
                     +np.einsum("bij,bj->bi",k,y[:,t,26])
                     +np.einsum("bij,bj->bi",g,packet26))
    mse=((states-x)**2)[:,:,2:].sum(axis=2)
    if fixture=="health":np.testing.assert_allclose(mse[:,20:].mean(1),archive[tag+"S_fixed_table_T_losses"][:,5],rtol=0,atol=2e-6)
    else:np.testing.assert_allclose(mse[:,50:110].mean(1),archive[tag+"S_fixed_table_T_losses"][:,0],rtol=0,atol=2e-6)
    return innov,keys,gates[:,:,6],delta[:,:,6]


def samples(trace,fixture,health=False):
    r,m,g,d=trace;rows=[];n=len(g);runs=np.zeros((n,150),int)
    for t in range(1,150):runs[:,t]=np.where(g[:,t],runs[:,t-1]+1,0)
    ticks=range(20,150) if health else range(START,STOP)
    for p in range(n):
        for t in ticks:
            scored=(20<=t<150) if health else (START<=t<STOP)
            active=("health" if health else
                    "active" if scored and (fixture=="continuous" or (t-START)%10<6) else
                    "inactive" if scored else "outside")
            past=r[p,max(0,t-L):t]
            mats=m[p,max(0,t-L):t]
            raw=np.zeros((L,2));keyed=np.zeros((L,2))
            raw[-len(past):]=past
            if len(past):keyed[-len(past):]=np.einsum("tji,tj->ti",mats,past)
            rows.append({"fixture":fixture,"path":p,"tick":t,"run":runs[p,t],
                         "alarm":bool(g[p,t]),"phase":active,"scored":scored,"raw":raw.reshape(-1),
                         "key":keyed.reshape(-1),"M":m[p,t],"delta":d[p,t],
                         "latent":m[p,t].T@d[p,t]})
    return rows


def ridge(X,Y,alpha=1.):
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1.
    Z=(X-mu)/sd;X1=np.column_stack((Z,np.ones(len(Z))))
    reg=np.eye(X1.shape[1])*alpha;reg[-1,-1]=0.
    coef=np.linalg.solve(X1.T@X1+reg,X1.T@Y)
    return mu,sd,coef


def apply(f,X):
    mu,sd,b=f
    return np.column_stack(((X-mu)/sd,np.ones(len(X))))@b


def fit_predict(train,test,feature,target,fold,test_phase):
    train_ids=[i for i,r in enumerate(train) if r["path"]%2!=fold and r["alarm"] and r["scored"]
               and r["tick"]>=START+L and r["phase"] in ("active","inactive")]
    val_ids=[i for i in train_ids if train[i]["path"]%4==(1 if fold==0 else 0)]
    fit_ids=[i for i in train_ids if i not in set(val_ids)]
    # The ridge grid is selected only on training paths, by inner path-group CV.
    losses=[]
    if val_ids:
        xv=np.stack([train[i][feature] for i in val_ids]);yv=np.stack([train[i][target] for i in val_ids])
        xf=np.stack([train[i][feature] for i in fit_ids]);yf=np.stack([train[i][target] for i in fit_ids])
        for alpha in ALPHAS:losses.append((float(np.mean((apply(ridge(xf,yf,alpha),xv)-yv)**2)),alpha))
        alpha=min(losses)[1]
    else:alpha=1.
    model=ridge(np.stack([train[i][feature] for i in train_ids]),
                np.stack([train[i][target] for i in train_ids]),alpha)
    out={}
    for i,row in enumerate(test):
        if row["path"]%2!=fold or not row["alarm"] or not row["scored"] or row["tick"]<START+L or row["phase"]!=test_phase:continue
        prediction=apply(model,row[feature][None])[0]
        if target=="latent":prediction=row["M"]@prediction
        out[(row["path"],row["tick"])]=prediction
    return out,{"ridge_alpha":alpha,"outer_train_alarm_samples":len(train_ids),
                "inner_validation_paths":len({train[i]["path"] for i in val_ids})}


def prior_predict(train,test,fold,test_phase):
    picked=[r for r in train if r["path"]%2!=fold and r["alarm"] and r["tick"]>=START+L
            and r["phase"] in ("active","inactive")]
    means={}
    for duration in (1,2,3,4):
        values=[r["M"].T@r["delta"] for r in picked if min(r["run"],4)==duration]
        means[duration]=np.mean(values,axis=0) if values else np.zeros(2)
    out={}
    for row in test:
        if row["path"]%2==fold and row["alarm"] and row["tick"]>=START+L and row["phase"]==test_phase:
            out[(row["path"],row["tick"])]=row["M"]@means[min(row["run"],4)]
    return out,means


def path_metrics(rows,preds,phase,fixture):
    result=[]
    for path in range(512):
        hits=[r for r in rows if r["fixture"]==fixture and r["path"]==path and r["phase"]==phase and r["alarm"]]
        for method in METHODS:
            changes=[];energies=[];coverage=0
            for row in hits:
                value=preds.get((method,fixture,path,row["tick"]),np.zeros(2)) if method!="zero" else np.zeros(2)
                coverage+=int(method!="zero" and (fixture,path,row["tick"]) in preds.get("_keys",set()))
                delta=row["delta"]
                changes.append(float(np.sum((delta-value)**2)-np.sum(delta**2)))
                energies.append(float(np.sum(value**2)))
            result.append({"fixture":fixture,"phase":phase,"path":path,"method":method,
                           "alarm_ticks":len(hits),"covered":coverage,
                           "error_change":float(np.mean(changes)) if changes else 0.,
                           "correction_energy":float(np.mean(energies)) if energies else 0.})
    return result


def summarize(rows):
    rng=np.random.default_rng(np.random.SeedSequence(SEED));out=[]
    for fixture,phase in (("continuous","active"),("burst","active"),("burst","inactive"),("health","health")):
        sub=[r for r in rows if r["fixture"]==fixture and r["phase"]==phase]
        for method in METHODS:
            vals=np.array([r["error_change"] for r in sub if r["method"]==method])
            cover=np.array([r["covered"]/max(r["alarm_ticks"],1) for r in sub if r["method"]==method])
            energy=np.array([r["correction_energy"] for r in sub if r["method"]==method])
            def ci(v):
                if len(v)==0:return (0.,0.,0.)
                means=[]
                for s in range(0,BOOT,500):
                    k=min(500,BOOT-s);w=rng.multinomial(len(v),np.full(len(v),1/len(v)),size=k)/len(v)
                    means.extend((w@v).tolist())
                return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))
            out.append({"fixture":fixture,"phase":phase,"method":method,"paths":len(vals),
                        "error_change":ci(vals),"coverage":ci(cover),"correction_energy":ci(energy)})
        baseline={r["path"]:r["error_change"] for r in sub if r["method"]=="history_only"}
        keyed={r["path"]:r["error_change"] for r in sub if r["method"]=="key_history"}
        paired=np.array([keyed[i]-baseline[i] for i in sorted(set(keyed)&set(baseline))])
        # Paired path bootstrap for the marginal value of the explicit key sequence.
        means=[]
        for start in range(0,BOOT,500):
            k=min(500,BOOT-start);w=rng.multinomial(len(paired),np.full(len(paired),1/len(paired)),size=k)/len(paired)
            means.extend((w@paired).tolist())
        out.append({"fixture":fixture,"phase":phase,"method":"key_minus_history_only",
                    "paths":len(paired),"error_change":(float(paired.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))),
                    "coverage":None,"correction_energy":None})
        baseline={r["path"]:r["error_change"] for r in sub if r["method"]=="key_mean_prior"}
        keyed={r["path"]:r["error_change"] for r in sub if r["method"]=="key_history"}
        paired=np.array([keyed[i]-baseline[i] for i in sorted(set(keyed)&set(baseline))])
        means=[]
        for start in range(0,BOOT,500):
            k=min(500,BOOT-start);w=rng.multinomial(len(paired),np.full(len(paired),1/len(paired)),size=k)/len(paired)
            means.extend((w@paired).tolist())
        out.append({"fixture":fixture,"phase":phase,"method":"key_history_minus_key_mean_prior",
                    "paths":len(paired),"error_change":(float(paired.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))),
                    "coverage":None,"correction_energy":None})
    return out


def main():
    model=task3.audit_frontend.load_model(INDEX)
    attack_traces={};attack_rows=[]
    for fixture in ("continuous","burst"):
        with np.load(ARCH/f"{CONFIG}_{fixture}_attack.npz") as a:
            trace=replay_T(a,"no_repeat_",fixture,model)
        attack_traces[fixture]=trace;attack_rows.extend(samples(trace,fixture))
    with np.load(ARCH/f"{CONFIG}_continuous_health.npz") as a:
        trace_health=replay_T(a,"no_repeat_","health",model)
    health_rows=samples(trace_health,"health",True)
    predictions={};training_meta=[]
    for fold in (0,1):
        for fixture in ("continuous","burst"):
            train=[r for r in attack_rows if r["fixture"]==fixture]
            test=[r for r in attack_rows if r["fixture"]==fixture]
            for method,feature,target in (("history_only","raw","delta"),("key_history","key","latent")):
                for test_phase in (("active","inactive") if fixture=="burst" else ("active",)):
                    result,meta=fit_predict(train,test,feature,target,fold,test_phase)
                    predictions.update({(method,fixture,p,t):v for (p,t),v in result.items()})
                    training_meta.append({"fold":fold,"fixture":fixture,"method":method,
                                          "evaluation_phase":test_phase,**meta})
            for test_phase in (("active","inactive") if fixture=="burst" else ("active",)):
                prior,_=prior_predict(train,test,fold,test_phase)
                predictions.update({("key_mean_prior",fixture,p,t):v for (p,t),v in prior.items()})
        # Apply attack-fitted models to independent health paths.
        for method,feature,target in (("history_only","raw","delta"),("key_history","key","latent")):
            result,meta=fit_predict(attack_rows,health_rows,feature,target,fold,"health")
            predictions.update({(method,"health",p,t):v for (p,t),v in result.items()})
            training_meta.append({"fold":fold,"fixture":"attack_to_health","method":method,**meta})
        prior,_=prior_predict(attack_rows,health_rows,fold,"health")
        predictions.update({("key_mean_prior","health",p,t):v for (p,t),v in prior.items()})
    result_rows=[]
    for fixture,rows,phase_names in (("continuous",attack_rows,("active",)),
                                     ("burst",attack_rows,("active","inactive")),
                                     ("health",health_rows,("false_alarm",))):
        phase_names=("health",) if fixture=="health" else phase_names
        for phase in phase_names:
            source_phase=("health" if fixture=="health" else phase)
            for path in range(512):
                hits=[r for r in rows if r["fixture"]==fixture and r["path"]==path and r["phase"]==source_phase and r["alarm"]]
                for method in METHODS:
                    changes=[];energies=[];covered=0
                    for row in hits:
                        key=(method,fixture,path,row["tick"])
                        value=predictions.get(key,np.zeros(2)) if method!="zero" else np.zeros(2)
                        covered+=int(method!="zero" and key in predictions)
                        delta=row["delta"]
                        changes.append(float(np.sum((delta-value)**2)-np.sum(delta**2)))
                        energies.append(float(np.sum(value**2)))
                    result_rows.append({"fixture":fixture,"phase":phase,"path":path,"method":method,
                                        "alarm_ticks":len(hits),"covered":covered,
                                        "error_change":float(np.mean(changes)) if changes else 0.,
                                        "correction_energy":float(np.mean(energies)) if energies else 0.})
    summary=summarize(result_rows)
    OUT.mkdir(exist_ok=True)
    for name,rows in (("KEY_PREDICTABILITY_PATHS.csv",result_rows),("KEY_PREDICTABILITY.csv",summary)):
        with (OUT/name).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    report={"status":"complete_cross_fitted_one_step_predictability_development",
            "configuration":CONFIG,"folding":"Path-ID parity within fixture; train paths and test paths are disjoint per fold.",
            "history_reports":L,"ridge_alpha_grid":ALPHAS,"fit":training_meta,
            "alarm_only_coverage_in_denominator":True,
            "health_true_disturbance_is_zero":True,
            "current_report_excluded_from_forecast_features":True,
            "current_key_used_only_after_report_arrival_for_key_inverse-transform":True,
            "closed_loop_compensation_run":False,
            "task4_RUN_sha256":sha(ARCH/"RUN.json"),
            "protocol_sha256":sha(OUT/"KEY_AIDED_PREDICTABILITY_PROTOCOL.md"),
            "interpretation":"Post-hoc development on already viewed Task-4 namespace-85 paths; one-step residual prediction only, no compensation closed loop, no new independent confirmation.",
            "pathwise_sha256":sha(OUT/"KEY_PREDICTABILITY_PATHS.csv"),
            "summary_sha256":sha(OUT/"KEY_PREDICTABILITY.csv")}
    (OUT/"KEY_PREDICTABILITY.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":report["status"],"summary":summary},indent=2))


if __name__=="__main__":main()
