"""Independent small-sample raw-innovation replay and full stored-array checks for A-prime."""
import json
import numpy as np

from analyze_audit import Data
from audit_replay import healthy, matrices, modes, Q
from scale_matching_audit import DEST, OUT, PAIRS, digest, write_json


def literal(data, path_id, model, m, legal, fitted, phi, scale):
    idx=path_id
    x,y,s,other,delta,normalized=(data[k][idx] for k in ("x","local","sender","other","delta","score"))
    run=0;loss=np.zeros((6,3)); eligible=np.zeros(2,int)
    shared_stats=np.zeros((len(PAIRS),5))
    state=np.zeros((6,4)); sigma=np.repeat(m.P[None],6,axis=0)
    words=np.full(6,7**6-1)
    sender=int(model.links[-1,1])
    h_others=sum(model.H[n] for n in model.neighbors[26] if n!=sender)
    upper=np.asarray(fitted["upper"])
    terms=((0,0.),(1,0.),(0,scale),(1,1.),(0,scale*phi),(1,phi))
    for tick in range(1,150):
        alarm=normalized[tick]>1.
        run=run+1 if alarm else 0
        a=np.full(6,6)
        if alarm:
            unit=3*int(np.searchsorted(fitted["boundaries"],normalized[tick],side="right"))
            cell=fitted["cells"][unit+int(run>=2)+int(run>=4)]
            # Evaluate each candidate's T_q and trace directly (the fast audit uses the adjoint).
            h=np.empty((6,7))
            for policy in range(6):
                trial=m.F@sigma[policy]@m.F.transpose(0,2,1)+m.O
                h[policy]=np.einsum("ab,qba->q",m.gram,trial)
            h-=h[:,[-1]]
            common=np.zeros(6,int)
            for policy,(label,coefficient) in enumerate(terms):
                for group,history,cost,output in ((policy,words[policy],h[policy],a),
                                                  (0,words[0],h[0],common)):
                    candidates=np.flatnonzero(legal[history])
                    value=upper[cell,label,candidates]+coefficient*cost[candidates]
                    output[policy]=candidates[value<=np.min(value)+1e-10][-1]
            if 50<=tick<=90:
                eligible[0]+=1
                if legal[words[0]].sum()>=2:
                    eligible[1]+=1
                    for p,(i,j) in enumerate(PAIRS):
                        if common[i]!=common[j]:
                            dh=h[0,common[j]]-h[0,common[i]]
                            shared_stats[p]+=[1,dh,dh < -1e-10,abs(dh)<=1e-10,dh>1e-10]
        prediction=state@model.A.T
        own=y[tick]-prediction@model.H[26].T
        remote=other[tick]-prediction@h_others.T
        remote+=Q[a,None]*(s[tick]+delta[tick]-prediction@model.H[sender].T)
        state=prediction+np.einsum("pij,pj->pi",m.K[a],own)+np.einsum("pij,pj->pi",m.G[a],remote)
        sigma=m.F[a]@sigma@m.F[a].transpose(0,2,1)+m.O[a]
        words=(words*7+a)%7**6
        err=((state[:,2:]-x[tick,2:])**2).sum(1)
        for p,(begin,end) in enumerate(((50,110),(110,150),(20,150))):
            if begin<=tick<end:loss[:,p]+=err/(end-begin)
    return loss,eligible,shared_stats


def main():
    manifest=json.loads((DEST/"manifest.json").read_text())
    worst=0.; health_error=0.; path_count=0
    with np.load(modes.OUTPUT/"mode_tables.npz") as gains:
        for scene, spec in enumerate(healthy.scenarios()):
            row=json.loads((OUT/f"decisions_{scene:02}.json").read_text())
            model=healthy.setup(spec,gains);m=matrices(gains,spec,model)
            key=f"{spec[1]}_{spec[0]}_target{spec[2]}_qwords.npz"
            with np.load(OUT/"graphs"/key) as z:legal=z["legal"]
            for condition in row["conditions"]:
                source,fixture=condition.split("_");data=Data(scene,source,fixture)
                path=DEST/f"{scene:02}_{condition}.npz"
                assert digest(path)==manifest["path_outputs_sha256"][path.name]
                with np.load(path) as z:
                    n=z["counts"]; al=z["action"]; saved=z["loss"]
                    assert (al[:,:,0]<=n[:,1,None]).all()
                    np.testing.assert_array_equal(al[:,:,0],al[:,:,2:].sum(2))
                    with np.load(OUT/f"path_losses_{scene:02}.npz") as old:
                        np.testing.assert_array_equal(saved[:,[0,1,3,5]],old[condition][:,[0,3,4,5],1:])
                    # Include a path that actually differentiates a scale pair if one exists.
                    selection=sorted(set((0,1023,int(np.argmax(al[:,:2,0].sum(1))))))
                    for i in selection:
                        loss,count,stats=literal(data,i+(0 if source=="D0" else 1024),model,m,legal,
                                                 row["table"],row["phi"],manifest["scale"]["applied"])
                        np.testing.assert_allclose(loss,saved[i],rtol=2e-9,atol=2e-8)
                        np.testing.assert_array_equal(count,n[i])
                        np.testing.assert_array_equal(stats[:,[0,2,3,4]],al[i,:,[0,2,3,4]].T)
                        np.testing.assert_allclose(stats[:,1],al[i,:,1],rtol=2e-10,atol=2e-11)
                        worst=max(worst,float(np.abs(loss-saved[i]).max()))
                        health_error=max(health_error,float(np.abs(stats[:,1]-al[i,:,1]).max()))
                        path_count+=1
    result=dict(conditions=len(manifest["checks"]),path_condition_checks=path_count,
                max_abs_literal_MSE_difference=worst,max_abs_same_state_H_sum_difference=health_error,
                historical_four_policy_path_arrays="bitwise identical for all 44 conditions")
    write_json(DEST/"verification.json",result)
    print(result)


if __name__=="__main__":
    main()
