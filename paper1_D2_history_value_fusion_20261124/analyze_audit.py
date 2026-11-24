"""Historical action audit. All decisions below are offline, on frozen path inputs."""
import argparse
import json
import numpy as np
from scipy.stats import spearmanr, t
from audit_graph import Q, HISTORIES
from audit_replay import CACHE, OUT, D1, modes, healthy, matrices, mv

CUT = 41
PAIRS = ((0,1),(1,2),(0,2))
PERIODS = ("common","window","tail","post20")
IDS = np.arange(7)


class Data:
    def __init__(self, scene, source, fixture):
        self.folder = CACHE/f"{scene}_{source}_{fixture}"

    def __getitem__(self,key):
        return np.load(self.folder/(key+".npy"),mmap_mode="r")


def classes(s, run, edges):
    score=np.searchsorted(edges,s,side="right")
    return 3*score+np.where(run==1,0,np.where(run<=3,1,2)),9+score


def table(fx,scene):
    boundary=np.quantile(np.concatenate([b["s"][:512,:CUT][b["run"][:512,:CUT]>0] for b in fx]),[1/3,2/3])
    counts=[]; totals=[]
    for b in fx:
        s=b["s"][512:1024,:CUT];r=b["run"][512:1024,:CUT];child,parent=classes(s,r,boundary)
        labels=np.stack([b[key][512:1024,:CUT] for key in ("Y1","Y60")],axis=-2)
        nb=[];sb=[]
        for c in range(13):
            gate=(r>0) & ((child==c) if c<9 else ((parent==c) if c<12 else True))
            nb.append(gate.sum(1))
            sb.append(np.einsum("pt,ptlq->plq",gate,labels))
        counts.append(np.stack(nb,1));totals.append(np.stack(sb,1))
    n=np.stack(counts);s=np.stack(totals)
    den=n.sum((0,1))
    mean=np.divide(s.sum((0,1)),den[:,None,None],out=np.zeros((13,2,7)),where=den[:,None,None]>0)
    infl=(s-n[:,:,:,None,None]*mean)/np.maximum(1,den)[:,None,None]
    infl=(infl-infl.mean(1,keepdims=True))*np.sqrt(512/511)
    se=np.sqrt((infl**2).sum((0,1)))
    support=(n>0).sum((0,1))
    usable=(support>=30)&(se[:,:,1:]>0).all((1,2))
    assert usable[12]
    rng=np.random.default_rng([20261124,6604,scene])
    dist=(rng.normal(size=(4096,1024))@infl.reshape(1024,-1)).reshape(4096,13,2,7)
    standardized=-dist/np.where(se>0,se,1)
    critical=float(np.quantile(standardized[:,usable,:,1:].max((1,2,3)),.95))
    ucb=mean+critical*se
    assert np.all(ucb[:,:,[0]]==0)
    lookup=np.array([i if usable[i] else (9+i//3 if usable[9+i//3] else 12) for i in range(9)])
    info=dict(boundaries=boundary.tolist(),support=support.tolist(),events=den.tolist(),cells=lookup.tolist(),
              critical=critical,mean=mean.tolist(),se=se.tolist(),upper=ucb.tolist())
    return boundary,lookup,mean,ucb,info


def argmin(cost,valid):
    eligible=np.where(valid,cost,np.inf)
    near=eligible<=eligible.min(-1,keepdims=True)+1e-10
    assert near.any(-1).all()
    return 6-near[...,::-1].argmax(-1)


def ci(x,alpha=.05):
    x=np.asarray(x,dtype=float)
    v=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(len(x))
    half=t.ppf(1-alpha/2,len(x)-1)*se
    return np.stack([v,v-half,v+half],axis=-1).tolist()


def ordering(pred,truth):
    i=int(pred.argmin())
    return dict(rho=float(spearmanr(pred,truth).statistic) if np.ptp(pred)>0 and np.ptp(truth)>0 else None,
                predicted_q=float(Q[i]),observed_q=float(Q[truth.argmin()]),
                excess=float(truth[i]-truth.min()),predicted=pred.tolist(),closed=truth.tolist())


def common_audit(data, sl, pars, phi):
    cuts,lookup,means,upper,_=pars
    s=data["s"][sl,:CUT];run=data["run"][sl,:CUT]
    cell=lookup[classes(s,run,cuts)[0]]
    valid=(data["mask"][sl,:CUT,None] & (1<<IDS))!=0
    alarm=run>0
    free=alarm & (valid.sum(2)>=2)
    values={k:data[k][sl,:CUT] for k in ("Y1","Y60","H","loss1","actual")}
    result={};deltas={}
    for label,ki in enumerate(("Y1","Y60")):
        predicted=means[cell,label]
        risk=upper[cell,label]
        actions=[argmin(risk+weight*values["H"],valid) for weight in (0.,1.,phi)]
        pairs={}
        for i,j in PAIRS:
            a,b=actions[i],actions[j]
            flip=free & (a!=b)
            samples=int(flip.sum())
            damage={}
            for k in ("Y1","Y60","H","loss1"):
                v=values[k]
                dif=(np.take_along_axis(v,b[...,None],2)-np.take_along_axis(v,a[...,None],2))[...,0]
                damage[k]=float(dif[flip].mean()) if samples else None
            name=f"{i}_{j}"
            pairs[name]=dict(n_alarm=int(alarm.sum()),singleton=int((alarm & ~free).sum()),
                             eligible=int(free.sum()),flips=samples,rate=samples/free.sum() if free.any() else None,
                             on_flips=damage)
        # Same origin and end mask for all grid values, including fixed-policy comparisons.
        pred=predicted[alarm].mean(0)
        measured=values[ki][alarm].mean(0)
        actual=values["actual"][alarm].mean(0)
        full=data["fixed"][sl].mean(0)
        full_alarm=data["run"][sl]>0
        extra={}
        if ki=="Y1":
            full_labels=data["Y1"][sl][full_alarm].mean(0)
            extra["all_origins_single_tick_to_full_window"]=ordering(full_labels,full)
        result[ki]=dict(pairs=pairs,table_labels=ordering(pred,measured),**extra,
                        table_to_closed_origin=ordering(pred,actual),
                        label_to_closed_origin=ordering(measured,actual),
                        label_to_full_window=ordering(measured,full),
                        relative_prediction_mae=float(np.abs((pred-pred[-1])-(measured-measured[-1])).mean()))
    return result


def replay(data,sl,m,legal,pars,phi,source):
    # Apply each offline counterfactual independently to the entire persisted path.
    cuts,lookup,_,upper,_=pars
    x,local,other,sender,delta,score=(data[k][sl] for k in ("x","local","other","sender","delta","score"))
    alarm=score>1
    n=len(x);policies=6
    state=np.zeros((n,policies,4));cov=np.broadcast_to(m.P,(n,policies,4,4)).copy()
    hist=np.full((n,policies),HISTORIES-1);rr=np.zeros(n,int)
    loss=np.zeros((n,policies,4));actions=np.zeros((n,policies,7),int)
    multipliers=np.array([0.,1.,phi]*2)
    # Additional same-state audit on the histories induced by R. The state
    # estimate used for the offline label remains the frozen A reference.
    comparisons=[(label,i,j) for label in range(2) for i,j in PAIRS]
    shared=np.zeros((n,len(comparisons),7))
    actual_diff=np.zeros((n,len(comparisons)),int)
    label_arrays={key:data[key][sl] for key in ("Y1","Y60","loss1")}
    ftr=m.F.transpose(0,2,1)
    # Linear functional of prior covariance for each candidate; compare to the q=1 reference.
    adj=ftr @ m.gram @ m.F
    beta=np.einsum("ij,qji->q",m.gram,m.O)
    for k in range(1,150):
        g=alarm[:,k];rr=(rr+1)*g
        a=np.full((n,policies),6,dtype=int)
        if g.any():
            cell=lookup[classes(score[g,k],rr[g],cuts)[0]]
            risk=upper[cell][:,[0,0,0,1,1,1],:]
            h=np.einsum("qij,bpji->bpq",adj,cov[g])+beta
            objective=risk+multipliers[:,None]*(h-h[:,:,-1:])
            a[g]=argmin(objective,legal[hist[g]])
            if 50<=k<91:
                t0=k-50
                for v,(label,i,j) in enumerate(comparisons):
                    ref=3*label
                    valid=legal[hist[g,ref]]
                    h0=h[:,ref,:]-h[:,ref,-1:]
                    r0=risk[:,ref,:]
                    choices=[argmin(r0+v0*h0,valid) for v0 in (0.,1.,phi)]
                    ai,aj=choices[i],choices[j]
                    changed=ai!=aj
                    row=np.arange(len(ai))
                    shared[g,v,0]+=1
                    shared[g,v,1]+=valid.sum(1)>=2
                    shared[g,v,2]+=changed
                    for col,key in enumerate(("Y1","Y60","loss1"),3):
                        val=label_arrays[key][g,t0]
                        shared[g,v,col]+=val[row,aj]-val[row,ai]
                    shared[g,v,6]+=h0[row,aj]-h0[row,ai]
            if 50<=k<110:
                for v,(label,i,j) in enumerate(comparisons):
                    actual_diff[g,v]+=a[g,3*label+i]!=a[g,3*label+j]
        assert legal[hist,a].all()
        hist=(hist*7+a)%HISTORIES
        f=m.F[a]
        packet=other[:,k,None]+Q[a][...,None]*(sender[:,k,None]+delta[:,k,None])
        state=mv(f,state)+mv(m.K[a],local[:,k,None])+mv(m.G[a],packet)
        cov=f@cov@f.swapaxes(-1,-2)+m.O[a]
        sq=((state[:,:,2:]-x[:,k,None,2:])**2).sum(2)
        if k>=20:loss[:,:,3]+=sq/130
        if k>=110:loss[:,:,2]+=sq/40
        if 50<=k<110:
            loss[:,:,1]+=sq/60
            for p in range(6):
                np.add.at(actions[:,p],(np.arange(n),a[:,p]),g.astype(int))
            if k<91:loss[:,:,0]+=sq/41
    differences={}
    for offset,key in [(0,"Y1"),(3,"Y60")]:
        differences[key]={}
        for i,j in PAIRS:
            differences[key][f"{i}_{j}"]=ci(loss[:,offset+j]-loss[:,offset+i])
    sr={}
    for v,(label,i,j) in enumerate(comparisons):
        totals=shared[:,v].sum(0)
        sr[f"{label}_{i}_{j}"]=dict(alarm=int(totals[0]),eligible=int(totals[1]),flips=int(totals[2]),
            sums_Y1_Y60_observed_H=totals[3:].tolist(),own_history_full_window_flips=int(actual_diff[:,v].sum()))
    return loss,dict(path_count=n,periods=PERIODS,policy_order=["R","RH","RHV"]*2,common_R_history=sr,
                     mean=loss.mean(0).tolist(),paired=differences,
                     alarm_action_counts=actions.sum(0).tolist())


def main(scene):
    spec=healthy.scenarios()[scene];dyn,top,sender,_=spec
    with np.load(modes.OUTPUT/"mode_tables.npz") as gains:
        model=healthy.setup(spec,gains);m=matrices(gains,spec,model)
    with np.load(OUT/"graphs"/f"{top}_{dyn}_target{sender}_qwords.npz") as words: legal=words["legal"]
    kappa=json.loads((D1/"results/D1A_link_value.json").read_text())["models"][f"{top}_{dyn}"]["link_scans"][str(sender)]["kappa"]["keep"]["60"]
    phi=1+max(0,kappa)/(1+max(0,kappa))
    fx=[Data(scene,"B2",f) for f in ("continuous","burst")]
    pars=table(fx,scene)
    report=dict(scene=scene,spec=spec,kappa=kappa,phi=phi,table=pars[-1],conditions={})
    arrays={}
    for source in ["B2","B1"]+(["D0"] if top=="U" else []):
        for fixture in (["healthy"] if source=="B1" else ["continuous","burst"]):
            data=Data(scene,source,fixture)
            sl=slice(0,None) if source=="D0" else slice(1024,None)
            key=source+"_"+fixture
            losses,record=replay(data,sl,m,legal,pars,phi,source)
            arrays[key]=losses
            if source!="B1":
                record["common_prior"]=common_audit(data,sl,pars,phi)
                score=data["s"][sl,:CUT];run=data["run"][sl,:CUT]
                cells=classes(score,run,pars[0])[0]
                v=data["actual"][sl,:CUT,4]
                y=data["Y1"][sl,:CUT,4]
                record["strata"]=[dict(cell=c,n=int(mask.sum()),mean_missing_q08=float(v[mask].mean()),
                    mean_Y1_q08=float(y[mask].mean()))
                    for c in range(9) if (mask:=((cells==c)&(run>0))).any()]
            else:arrays["B1_reference"]=data["fixed"][sl,-1].copy()
            report["conditions"][key]=record
    (OUT/f"decisions_{scene:02}.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(OUT/f"path_losses_{scene:02}.npz",**arrays)
    print("audited scene",scene,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("scenes",nargs="+",type=int,choices=range(12))
    for scene in p.parse_args().scenes:main(scene)
