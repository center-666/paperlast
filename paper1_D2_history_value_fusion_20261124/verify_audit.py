"""Recheck archived decisions from raw innovations and single-word observations."""
import json
from pathlib import Path
import hashlib
import numpy as np
from audit_replay import D1, OUT, REF, Q, modes, healthy, read_paths, matrices

def literal(path,m,model,legal,table,phi):
    n=min(4,len(path.x))
    results=[]
    for label in range(2):
        for factor in (0.,1.,phi):
            state=np.zeros((n,4));P=np.repeat(m.P[None],n,axis=0)
            history=np.full(n,7**6-1);run=np.zeros(n,int);loss=np.zeros((n,4))
            for k in range(1,150):
                alarm=path.alarm[:n,k];run=np.where(alarm,run+1,0)
                a=np.full(n,6)
                for i in np.flatnonzero(alarm):
                    box=3*np.searchsorted(table['boundaries'],path.s[i,k],side='right')+min(2,(run[i]>1)*1+(run[i]>3)*1)
                    ucb=np.array(table['upper'][table['cells'][box]][label])
                    candidates=np.flatnonzero(legal[history[i]])
                    values=[]
                    for idx in candidates:
                        v=m.F[idx]@P[i]@m.F[idx].T+m.O[idx]
                        values.append(ucb[idx]+factor*np.sum(m.gram*v.T))
                    allowed= np.asarray(values)<=min(values)+1e-10
                    a[i]=candidates[allowed][-1]
                assert legal[history,a].all()
                pred=state@model.A.T
                j=int(model.links[6,1])
                Hother=np.sum([model.H[l] for l in model.neighbors[26] if l!=j],axis=0)
                local=path.local[:n,k]-pred@model.H[26].T
                neighbor=path.other[:n,k]-pred@Hother.T
                target=path.sender[:n,k]+path.delta[:n,k]-pred@model.H[j].T
                state=pred+np.einsum('bij,bj->bi',m.K[a],local)+np.einsum(
                    'bij,bj->bi',m.G[a],neighbor+Q[a,None]*target)
                P=m.F[a]@P@m.F[a].transpose(0,2,1)+m.O[a]
                history=(history*7+a)%7**6
                err=np.sum((state[:,2:]-path.x[:n,k,2:])**2,axis=1)
                for col,(lo,hi) in enumerate(((50,91),(50,110),(110,150),(20,150))):
                    if lo<=k<hi:loss[:,col]+=err/(hi-lo)
            results.append(loss)
    return np.stack(results,axis=1)

def main():
    worst=0.;ncheck=0;max_sv=0.
    with np.load(modes.OUTPUT/'mode_tables.npz') as gi:
        for scene,spec in enumerate(healthy.scenarios()):
            dyn,top,sender,wm=spec;m0=healthy.setup(spec,gi);m=matrices(gi,spec,m0)
            key=f"{top}_{dyn}_target{sender}_q"
            graph=OUT/'graphs'
            row=json.loads((OUT/f'decisions_{scene:02}.json').read_text())
            with np.load(graph/(key+'words.npz')) as words:
                legal=words['legal']
                if scene%2==0:
                    idx=np.arange(0,7**7,997)
                    idx=np.unique(np.r_[idx,0,7**7-1,np.argmax(np.where(legal.ravel(),words['norm'],-1))])
                    cs=np.array([gi[key+f'{q:g}_C'][64,26] for q in Q])
                    local_max=0.
                    for code in idx:
                        sequence=np.unravel_index(code,(7,)*7)
                        product=np.linalg.multi_dot([m.F[i] for i in reversed(sequence)])
                        obs=np.vstack([cs[i]@np.linalg.matrix_power(m0.A,s) for s,i in enumerate(sequence)])
                        sv=np.linalg.svd(obs,compute_uv=False)
                        local_max=max(local_max,abs(words['norm'][code]-np.linalg.norm(product,2)),
                                      abs(words['sigma_min'][code]-sv[-1]))
                        assert words['obs_rank'][code]==np.linalg.matrix_rank(obs,tol=1e-10)
                    assert local_max<1e-12
                    max_sv=max(max_sv,local_max)
            for source in ['B2','B1']+(['D0'] if top=='U' else []):
                for fixture in (['healthy'] if source=='B1' else ['continuous','burst']):
                    batch_index=0 if source=='D0' else 16
                    for batch,p in enumerate(read_paths(scene,spec,m0,fixture,source)):
                        if batch==batch_index:break
                    actual=literal(p,m,m0,legal,row['table'],row['phi'])
                    with np.load(OUT/f'path_losses_{scene:02}.npz') as z:
                        expected=z[source+'_'+fixture][:4]
                    np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-7)
                    worst=max(worst,float(np.max(abs(actual-expected))));ncheck+=len(actual)
    result=dict(verified_paths=ncheck,policies_per_path=6,periods=4,
                max_abs_path_MSE_error=worst,graph_norm_or_sigma_max_sample_difference=max_sv,
                scope="Independent innovation update, scoring and tie break using frozen fitted UCB and original D1 path generator; sampled graph SVD cross-check; no new physical paths.")
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=="__main__":
    main()
