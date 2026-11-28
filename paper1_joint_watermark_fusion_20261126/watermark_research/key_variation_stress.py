"""Direction/amplitude held-out stress for cross-fitted Task-4 bias predictors."""
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
import frontend
import models_quality as quality

sys.path.insert(0,str(OUT))
import key_prediction_dev as pred

RIDGE_GRID=(.001,.01,.1,1.,10.,100.)
SEED=[20261128,9521,80]
CONFIG="U_unstable_R35x2";INDEX=2
METHODS=("zero","history_only","key_history","key_mean_prior")
SCENARIOS=(("fit_direction_1x",(6.,0.)),
           ("heldout_direction_minus45",(6/np.sqrt(2),-6/np.sqrt(2))),
           ("heldout_direction_plus45",(6/np.sqrt(2),6/np.sqrt(2))),
           ("heldout_amplitude_half",(3.,0.)),
           ("heldout_amplitude_1p5",(9.,0.)))


def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def target_replay(archive,fixture,vector,model,ops,threshold):
    y=archive["y"];x=archive["x"];keys=archive["no_repeat_keys"]
    active=np.ones(60) if fixture=="continuous" else (np.arange(60)%10<6).astype(float)
    delta=np.zeros((len(x),150,7,2))
    delta[:,50:110,6]=np.einsum("btij,tj->bti",keys[:,50:110,6],active[:,None]*vector)
    packets=(y[:,:,model.links[:,1]]+delta)
    score=frontend.scores(packets,keys,ops)
    gates=score>threshold
    n=len(x);state=np.zeros((n,150,4));residual=np.zeros((n,150,2))
    actions=np.full((n,150),6,dtype=np.int8)
    history=np.full(n,7**6-1,dtype=np.int64)
    graph=task3.configure(INDEX)[1]
    raw=(y.swapaxes(-1,-2)@model.adjacency.T).swapaxes(-1,-2)
    lookup={tuple(map(int,e)):i for i,e in enumerate(model.links[:6])}
    sender=int(model.links[6,1]);neighbors=model.neighbors[26]
    for tick in range(1,150):
        prior_state=state[:,tick-1]@model.A.T
        residual[:,tick]=y[:,tick,sender]+delta[:,tick,6]-prior_state@model.H[sender].T
        alarm=gates[:,tick,6]
        legal=graph[history]
        requested=np.where(alarm,4,6)
        feasible=legal & (np.arange(7)[None,:]>=requested[:,None])
        action=feasible.argmax(axis=1)
        actions[:,tick]=action
        history=(history*7+action)%7**6
        received=raw[:,tick,26].copy()
        for src in neighbors:
            edge=lookup.get((26,int(src)))
            if edge is not None:
                received-=gates[:,tick,edge,None]*y[:,tick,int(src)]
        q=np.where(alarm,quality.Q[action],1.)
        packet=(received+delta[:,tick,6]
                +(q-1.)[:,None]*(y[:,tick,sender]+delta[:,tick,6]))
        mode=alarm.astype(int)
        f=model.target["F"][mode,action];k=model.target["K"][mode,action];g=model.target["G"][mode,action]
        state[:,tick]=(np.einsum("bij,bj->bi",f,state[:,tick-1])
                       +np.einsum("bij,bj->bi",k,y[:,tick,26])
                       +np.einsum("bij,bj->bi",g,packet))
    return {"residual":residual,"delta":delta[:,:,6],"keys":keys[:,:,6],
            "gates":gates[:,:,6],"actions":actions,"state":state,"x":x,
            "alarm_score":score[:,:,6]}


def variant_rows(trace,fixture,variant):
    rows=[];n=len(trace["x"]);runs=np.zeros((n,150),dtype=np.int16)
    for tick in range(1,150):runs[:,tick]=np.where(trace["gates"][:,tick],runs[:,tick-1]+1,0)
    for path in range(n):
        for tick in range(50,110):
            active=(fixture=="continuous" or (tick-50)%10<6)
            phase="active" if active else "inactive"
            begin=max(0,tick-8)
            raw=trace["residual"][path,begin:tick]
            matrices=trace["keys"][path,begin:tick]
            rx=np.zeros((8,2));kx=np.zeros((8,2))
            if len(raw):
                rx[-len(raw):]=raw
                kx[-len(raw):]=np.einsum("tji,tj->ti",matrices,raw)
            M=trace["keys"][path,tick]
            delta=trace["delta"][path,tick]
            rows.append({"fixture":fixture,"variant":variant,"path":path,"tick":tick,
                         "run":int(runs[path,tick]),"alarm":bool(trace["gates"][path,tick]),
                         "phase":phase,"scored":True,"raw":rx.reshape(-1),
                         "key":kx.reshape(-1),"M":M,"delta":delta,"latent":M.T@delta,
                         "score":float(trace["alarm_score"][path,tick])})
    return rows


def fit_predictor(rows,fold,feature,target):
    train=[r for r in rows if r["path"]%2!=fold and r["alarm"] and r["tick"]>=58]
    val=[r for r in train if r["path"]%4==(1 if fold==0 else 0)]
    fit=[r for r in train if r["path"]%4!=(1 if fold==0 else 0)]
    X=np.stack([r[feature] for r in fit]);Y=np.stack([r[target] for r in fit])
    Xv=np.stack([r[feature] for r in val]);Yv=np.stack([r[target] for r in val])
    choices=[]
    for alpha in RIDGE_GRID:
        model=pred.ridge(X,Y,alpha)
        choices.append((float(np.mean((pred.apply(model,Xv)-Yv)**2)),alpha))
    alpha=min(choices)[1]
    model=pred.ridge(np.stack([r[feature] for r in train]),
                     np.stack([r[target] for r in train]),alpha)
    metadata={"alpha":alpha,"train_alarm_ticks":len(train),
              "inner_validation_path_count":len({r["path"] for r in val})}
    return model,metadata


def predict_for(rows,fold,models):
    output={}
    train=[r for r in rows if r["path"]%2!=fold and r["alarm"] and r["tick"]>=58]
    means={}
    for duration in (1,2,3,4):
        values=[r["latent"] for r in train if min(r["run"],4)==duration]
        means[duration]=np.mean(values,axis=0) if values else np.zeros(2)
    for row in rows:
        if row["path"]%2!=fold or not row["alarm"] or row["tick"]<58:continue
        output["history_only",row["path"],row["tick"]]=pred.apply(models["history_only"],row["raw"][None])[0]
        latent=pred.apply(models["key_history"],row["key"][None])[0]
        output["key_history",row["path"],row["tick"]]=row["M"]@latent
        output["key_mean_prior",row["path"],row["tick"]]=row["M"]@means[min(row["run"],4)]
    return output


def main():
    model=task3.audit_frontend.load_model(INDEX)
    operators=frontend.operators(model)
    threshold=task3.frontend_thresholds(INDEX)["no_repeat_matched"]
    source={fixture:np.load(ARCH/f"{CONFIG}_{fixture}_attack.npz") for fixture in ("continuous","burst")}
    base_rows={}
    base_models={}
    alpha_records=[]
    for fixture in ("continuous","burst"):
        trace=target_replay(source[fixture],fixture,np.asarray([6.,0.]),model,operators,threshold)
        base_rows[fixture]=variant_rows(trace,fixture,"fit_direction_1x")
        base_models[fixture]={}
        for fold in (0,1):
            for method,feature,target in (("history_only","raw","delta"),("key_history","key","latent")):
                fit,meta=fit_predictor(base_rows[fixture],fold,feature,target)
                base_models[fixture][fold,method]=fit
                alpha_records.append({"fixture":fixture,"fold":fold,"method":method,**meta})
    output_rows=[]
    scenario_path_means=[]
    for fixture in ("continuous","burst"):
        for variant,vector in SCENARIOS:
            trace=target_replay(source[fixture],fixture,np.asarray(vector),model,operators,threshold)
            if variant=="fit_direction_1x":
                original=np.load(ARCH/f"{CONFIG}_{fixture}_attack.npz")
                np.testing.assert_array_equal(trace["delta"],original["no_repeat_delta"][:,:,6])
                np.testing.assert_array_equal(trace["gates"],original["no_repeat_gates"][:,:,6].astype(bool))
                original.close()
            rows=variant_rows(trace,fixture,variant)
            for fold in (0,1):
                prediction=predict_for(rows,fold,{
                    "history_only":base_models[fixture][fold,"history_only"],
                    "key_history":base_models[fixture][fold,"key_history"]})
                # Reuse the exact registered mean-direction predictor fitted on train paths.
                train=[r for r in base_rows[fixture] if r["path"]%2!=fold and r["alarm"] and r["tick"]>=58]
                means={d:np.mean([r["latent"] for r in train if min(r["run"],4)==d],axis=0)
                       if any(min(r["run"],4)==d for r in train) else np.zeros(2) for d in (1,2,3,4)}
                for row in rows:
                    if row["path"]%2!=fold or not row["alarm"] or row["tick"]<58:continue
                    prediction["key_mean_prior",row["path"],row["tick"]]=row["M"]@means[min(row["run"],4)]
                for path in range(512):
                    selected=[r for r in rows if r["path"]==path and r["alarm"]]
                    errors=[]
                    for method in ("zero","history_only","key_history","key_mean_prior"):
                        vals=[]
                        for row in selected:
                            estimate=prediction.get((method,path,row["tick"]),np.zeros(2)) if method!="zero" else np.zeros(2)
                            d=row["delta"]
                            vals.append(float(np.sum((d-estimate)**2)-np.sum(d*d)))
                        errors.append(float(np.mean(vals)) if vals else 0.)
                    output_rows.append({"fixture":fixture,"variant":variant,"path":path,
                                        "fold":fold,"alarm_ticks":len(selected),
                                        "zero_change":errors[0],"history_change":errors[1],
                                        "key_change":errors[2],"key_prior_change":errors[3],
                                        "key_minus_history":errors[2]-errors[1],
                                        "key_minus_prior":errors[2]-errors[3]})
            print(fixture,variant,"done",flush=True)
    # Paired path bootstrap within fixture and variant, shared with all predictor arms.
    summary=[];rng=np.random.default_rng(np.random.SeedSequence([*SEED,91]))
    for fixture in ("continuous","burst"):
        for variant,_ in SCENARIOS:
            sub=[r for r in output_rows if r["fixture"]==fixture and r["variant"]==variant]
            for metric in ("zero_change","history_change","key_change","key_prior_change",
                           "key_minus_history","key_minus_prior"):
                vals=np.array([r[metric] for r in sub])
                draws=[]
                for start in range(0,20000,500):
                    n=min(500,20000-start);weights=rng.multinomial(len(vals),np.full(len(vals),1/len(vals)),size=n)/len(vals)
                    draws.extend((weights@vals).tolist())
                summary.append({"fixture":fixture,"variant":variant,"metric":metric,
                                "paths":len(vals),"alarm_paths":sum(r["alarm_ticks"]>0 for r in sub),
                                "mean_path_error_change":float(vals.mean()),
                                "lower95":float(np.quantile(draws,.025)),
                                "upper95":float(np.quantile(draws,.975)),
                                "mean_alarm_ticks":float(np.mean([r["alarm_ticks"] for r in sub]))})
    with (OUT/"KEY_VARIATION_PATHS.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(output_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(output_rows)
    with (OUT/"KEY_VARIATION_SUMMARY.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator="\n");writer.writeheader();writer.writerows(summary)
    result={"status":"complete_heldout_direction_amplitude_development",
            "fit_input_direction_and_scale":"[6,0], original Task-4 attack template",
            "heldout_variants":[{"name":n,"vector":list(v)} for n,v in SCENARIOS],
            "physical_world":"Existing Task-4 CRN paths reused for paired counterfactual waveform directions. This is a new input replay on already exposed physical paths, not an independent sample.",
            "detector_threshold":"Frozen Task-4 no-repeat matched threshold; never recalibrated by variant.",
            "predictor_feature_timing":"The forecast input only uses T-derived report residuals through k-1 and, for key models, past/current legal matrices after the new report arrives. The score itself still uses frozen full matched GLRT.",
            "ridge_fit":"Fit on base-direction paths from opposite path-ID parity. Ridge selection uses path-group internal validation within the training half.",
            "base_attack_sha256":{fixture:sha(ARCH/f"{CONFIG}_{fixture}_attack.npz") for fixture in ("continuous","burst")},
            "fit_source":"watermark_research/key_prediction_dev.py",
            "fit_source_sha256":sha(OUT/"key_prediction_dev.py") if (OUT/"key_prediction_dev.py").exists() else None,
            "crossfit_metadata":alpha_records,
            "outputs_sha256":{"paths":sha(OUT/"KEY_VARIATION_PATHS.csv"),"summary":sha(OUT/"KEY_VARIATION_SUMMARY.csv")},
            "code_sha256":sha(Path(__file__))}
    (OUT/"KEY_VARIATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"status":result["status"],"summary_rows":summary},indent=2))


if __name__=="__main__":main()
