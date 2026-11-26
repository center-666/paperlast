"""Direct SVD checks for the actual no-repeat receiver, with disjoint probe samples."""
import json
import numpy as np
import experiment as exp
import rotation_markov as markov
from matched_glrt import build_sender_geometry


def main():
    errors=[];checks=0;independence=[]
    for mi,model_name in enumerate(exp.MODEL):
        src,model,phy,ops,u,neg=markov.streams(3,0,mi,0,71,2)
        senders=model.links[:,1]
        raw=phy.y[:,:,senders]
        keys={"quarter_iid":markov.quarter_keys(u,False)[0][:,:,senders],
              "quarter_no_repeat":markov.quarter_keys(u,True)[0][:,:,senders],
              "negative":neg}
        probes=[markov.streams(2,0,mi,0,72,pop)[2].y[:,1,0] for pop in (0,1,2)]
        for a in range(3):
            for b in range(a):
                assert not np.array_equal(probes[a],probes[b])
        independence.append({"model":model_name,"population_roots":[list(markov.stream_root(0,mi,0,72,p)) for p in (0,1,2)]})
        for label,key in keys.items():
            reports=raw.copy()
            reports[:,50:110]+=(key[:,50:110]@src.rotated_raw_attack(model,0)[0][None,50:110,...,None])[...,0]
            actual=exp.receiver.matched(reports,key,ops)
            # Future observations and future keys are excluded from earlier scores.
            modified_y=reports.copy();modified_y[:,70:]+=1e4
            modified_k=key.copy();modified_k[:,70:]=2*np.eye(2)
            np.testing.assert_array_equal(exp.receiver.matched(modified_y,modified_k,ops)[:,:70],actual[:,:70])
            np.testing.assert_array_equal((key@np.swapaxes(key,-1,-2)),np.broadcast_to(np.eye(2),key.shape)) if label.startswith("quarter") else None
            for tick in (5,6,7,8,52,67,109,124):
                width=min(8,tick+1);first=tick-width+1;startup=first==0
                for e,sender in enumerate(senders):
                    geometry=build_sender_geometry(model,int(sender),startup,width)
                    blocks,ha=ops[width,startup]
                    packet=reports[0,first:tick+1,e].copy()
                    if startup:packet[0]=0
                    y=geometry.whitening_map@packet.ravel()
                    candidates=[]
                    for onset in range(int(startup),width):
                        contributions=blocks[e,onset:]@(key[0,first+onset:tick+1,e]@ha[e,onset:])
                        X=contributions.sum(axis=0)
                        bound=np.sum(np.linalg.norm(blocks[e,onset:],axis=(1,2))*
                                     np.linalg.norm(key[0,first+onset:tick+1,e]@ha[e,onset:],axis=(1,2)))
                        U,s,_=np.linalg.svd(X,full_matrices=False)
                        U=U[:,s>max(s[0]*1e-10,bound*1e-12)]
                        candidates.append(float(np.sum((U.T@y)**2)))
                    expected=max(candidates)
                    err=abs(actual[0,tick,e]-expected)
                    np.testing.assert_allclose(actual[0,tick,e],expected,atol=2e-7,rtol=2e-10)
                    errors.append(err);checks+=1
    exp.old.base._atomic_json(exp.HERE/"MARKOV_RECEIVER_AUDIT.json",{
        "cases":checks,"max_abs_score_vs_independent_svd":max(errors),
        "future_observation_and_key_invariance":True,"population_seed_independence":independence,
        "source_sha256":exp.old.base.digest(exp.HERE/"audit_markov_receiver.py")})
    print(checks,max(errors))


if __name__=="__main__":
    main()
