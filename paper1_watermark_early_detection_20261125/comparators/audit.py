"""Independent W0 readback, paired replay and deterministic W2 rank audit."""
import json
import numpy as np
import experiment as ex

old=ex.old
folder=ex.BASE/"attribution_confirm"


def readback():
    m=ex.read(folder/"RUN.json")
    arms="ABCDEF"
    results=[]
    for name in ex.MODEL:
        for scene in range(6):
            row=[]
            with np.load(folder/f"{name}_eval_{scene}.npz") as ar:
                for a in arms:
                    hit=ar[a+"_early_max"]>m["thresholds"][name][a][1]
                    np.testing.assert_array_equal(hit,ar[a+"_delay"]<3)
                    row.append([hit.mean(1).mean(),ar[a+"_attack_affected_mse"].mean(),
                                ar[a+"_postattack_affected_mse"].mean()])
            results.append(row)
    avg=np.mean(results,axis=0)
    np.testing.assert_allclose(avg[:4,0]*100,[64.06,65.39,90.19,91.09],atol=.005)
    return {"arm_order":list(arms),"columns":["early3","attack5_mse","recovery5_mse"],
            "means":avg.tolist(),"B-A":(avg[1]-avg[0]).tolist(),"D-C":(avg[3]-avg[2]).tolist(),
            "D-B":(avg[3]-avg[1]).tolist(),"C-E":(avg[2]-avg[4]).tolist(),"D-F":(avg[3]-avg[5]).tolist(),
            "old_thresholds":m["thresholds"]}


def replay(mi,scene):
    name=ex.MODEL[mi]
    manifest=ex.read(folder/"RUN.json")
    ns=tuple(manifest["identity"]["seed_namespace"])
    src,model,generate,estimate,ops=old.backend(name)
    onset=(50,65)[scene%2]
    rawfix=src.rotated_raw_attack(model,(0.,-45.,45.)[scene//2])[0][50:110]
    maxima=np.zeros(4)
    with np.load(folder/f"{name}_eval_{scene}.npz") as ar:
        for b in range(16):
            ix=slice(32*b,32*b+32)
            phy=generate(model,32,old.base.seed_word(ns,mi,scene,11,b,1))
            j=model.links[:,1]
            draws=old.short_draws(32,ns,mi,scene,12,b,j)
            for scheme,plugin,match in (("iid","A","C"),("negative_fixed","B","D")):
                key=old.matrices(draws,scheme)
                clean=phy.y[:,:,j]
                delta=np.zeros_like(clean)
                delta[:,onset:onset+60]=(key[:,onset:onset+60]@rawfix[None,...,None])[...,0]
                assert (delta[:,:onset]==0).all() and (delta[:,onset+60:]==0).all()
                for a in (plugin,match):
                    if a==plugin:
                        score=old.base.plugin_scores(model,phy,delta)
                        control=estimate(model,phy,delta,gate_on=False).scores
                    else:
                        score=old.matched(clean+delta,key,ops)
                        control=ex.receiver.matched(clean+delta,key,ops)
                    th=manifest["thresholds"][name][a][1]
                    gate=score>th
                    np.testing.assert_array_equal(gate,control>th)
                    maxima[0]=max(maxima[0],np.abs(score-control).max())
                    peak=score[:,onset:onset+3].max(1)
                    maxima[1]=max(maxima[1],abs(peak-ar[a+"_early_max"][ix]).max())
                    maxima[2]=max(maxima[2],abs(score.max((1,2))-ar[a+"_max"][ix]).max())
                    segment=gate[:,onset:onset+60]
                    np.testing.assert_array_equal(np.where(segment.any(1),segment.argmax(1),60),ar[a+"_delay"][ix])
                    post=estimate(model,phy,delta,gate_override=gate).post
                    error=((post-phy.x[:,:,None])**2).sum(-1)
                    for keyname,w in (("attack",slice(onset,onset+60)),("postattack",slice(onset+60,None)),("post20",slice(20,None))):
                        for nodes,v in (("affected",error[:,w][:,:,model.receiver_ids]),("network",error[:,w])):
                            maxima[3]=max(maxima[3],abs(v.mean((1,2))-ar[f"{a}_{keyname}_{nodes}_mse"][ix]).max())
    assert max(maxima)<2e-7
    return {"scene":scene,"paths":512,"four_arms_full_gate_disagreements":0,
            "errors_full_crosscode_early_archived_pathmax_archived_loss_archived":maxima.tolist()}


def geometry(mi,c):
    src,model,_,_,_=old.backend(ex.MODEL[mi])
    phy,draws,bits=ex.physical_and_keys(3,ex.namespace(0,mi,0,0,0),mi)
    raw=phy.y[:,:,model.links[:,1]]
    ops=ex.operators(mi,1.)
    rows={}
    for s in ex.SCHEMES:
        key=ex.schedule(s,draws,bits,c)
        clean=(key@np.linalg.solve(key,raw[...,None]))[...,0]
        scaled=np.max(abs(clean-raw)/np.maximum(1.,abs(raw)))
        assert scaled<1e-12
        # The online callable has only arrived reports, keys and public window maps.
        record={"onset":65,"reports":raw,"keys":key}
        apply=lambda rec: ex.receiver.matched(rec["reports"],rec["keys"],ops)
        scores=apply(record)
        del record["onset"]
        np.testing.assert_array_equal(apply(record),scores)
        future=key.copy(); future[:,66:]=2*np.eye(2)
        y=raw.copy();y[:,66:]=999.
        np.testing.assert_array_equal(ex.receiver.matched(y,future,ops)[:,:66],scores[:,:66])
        nullscore=ex.receiver.matched(raw,2.5*key,ops)
        nullerr=np.max(abs(nullscore-scores))
        assert nullerr<1e-7
        rows[s]={"roundtrip_scaled_error":float(scaled),"scalar_template_null_max_error":float(nullerr),
                 "time_alignment":[5,149],"key_shape":list(key.shape)}
        for m in (.98,1.,1.02):
            estimates=ex.receiver.matched(raw,key,ex.operators(mi,m))
            distances=[]; errors=[]; ranks=[]
            for t in (5,6,7,8,50,65,100,149):
                width=min(t+1,8); startup=t<8
                blocks,ha=ops[width,startup]
                for e in range(6):
                    W=blocks[e]
                    k=key[0,t-width+1:t+1,e]
                    y=raw[0,t-width+1:t+1,e].copy()
                    if startup:y[0]=0.
                    z=np.einsum("wri,wi->r",W,y)
                    vals=[]
                    for off in range(int(startup),width):
                        a=k[off:]@ha[e,off:]
                        WX=W[off:]@a
                        bound=(np.linalg.norm(W[off:],axis=(1,2))*np.linalg.norm(a,axis=(1,2)))
                        def projection(mult):
                            weights=mult**np.arange(len(a))
                            x=np.einsum("w,wri->ri",weights,WX)
                            U,sv,_=np.linalg.svd(x,full_matrices=False)
                            U=U[:,sv>max(1e-12*(weights@bound),sv[0]*1e-10)]
                            return U@U.T,U.shape[1]
                        P,r=projection(m); P0,_=projection(1.)
                        vals.append(z@P@z);ranks.append(r)
                        distances.append(np.linalg.norm(P-P0,2))
                    errors.append(abs(max(vals)-estimates[0,t,e]))
            assert max(errors)<2e-7,(s,m,max(errors))
            rows[s][str(m)]={"svd_max_score_error":max(errors),"rank_counts":np.bincount(ranks,minlength=3).tolist(),
                "projector_distance_p50_p95_max":np.quantile(distances,[.5,.95,1]).tolist(),
                "max_score_change":np.max(abs(estimates-scores))}
            if m!=1 and s in ("iid","negative"): assert max(distances)>.001
        if s=="none":assert rows[s]["1.0"]["rank_counts"][0]>0
    return rows


if __name__=="__main__":
    ex.receiver.kernel()
    results={"readback":readback(),"replay":{},"geometry":{}}
    for mi,name in enumerate(ex.MODEL):
        results["replay"][name]=replay(mi,mi)
        results["geometry"][name]=geometry(mi,ex.read(ex.HERE/"ENERGY_LOCK.json")[name]["encode_multiplier"])
    old.base._atomic_json(ex.HERE/"W0_W2_AUDIT.json",results)
