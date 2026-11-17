"""Causal attack policies. API never receives state, local data, M, or alarms."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.integrate import quad
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'common_g_reference_revision_20261115'))
import common_g_reference as ref
CFG=json.loads((H/'registration.json').read_text('utf8'))

def inverse_moments():
    # beta~N(0,.5I2) => ||beta||^2~Exp(mean1).
    lam=.5;a=1/lam
    ed=quad(lambda r:np.exp(-r)/(lam+r),0,np.inf,epsabs=1e-12)[0]
    ed2=quad(lambda r:np.exp(-r)/(lam+r)**2,0,np.inf,epsabs=1e-12)[0]
    eb=ed-a;eb2=ed2-2*a*ed+a*a
    return dict(mean=a+eb/2,coef_X=a*a+a*eb+eb2/4,coef_trace=eb2/8,
                mean_square=(a*a+ed2)/2)

def causal_policy(q,A,Hj,policy,Ta=7,L=10,E=10.):
    """q.shape=(N,T,m); each time reads only the allowed prefix.

    Remaining-energy projection applies to TOTAL attack, including the
    autonomous term. For theta0 and feasible autonomous input it is inactive.
    """
    N,T,m=q.shape;xi=np.asarray(policy['xi']);theta=policy['theta']
    family=policy['family'];lag=Hj@A@np.linalg.inv(Hj)/inverse_moments()['mean']
    # Roundoff reserve, common to every class. The declared physical upper
    # budget remains E; A0 already has a larger 2e-12 radial energy margin.
    executable_budget=E*(1-1e-12)
    remaining=np.full(N,executable_budget);a=np.zeros_like(q);clipped=np.zeros((N,T),bool)
    used_budget=np.zeros((N,T))
    for k in range(T):
        if Ta<=k<Ta+L:
            if policy.get('kind')=='public_first_packet':
                if k==Ta:a[:,k]=np.asarray(policy['packet'])
            elif policy.get('kind')=='feasible_reflection':
                v=-2*q[:,k];cost=np.einsum('ni,ni->n',v,v)
                allowed=cost<=remaining*(1-1e-12)
                a[:,k]=v*allowed[:,None]
            else:
                base=Hj@np.linalg.matrix_power(A,k-Ta)@xi
                feedback=np.zeros((N,m))
                if theta and family=='A1-H' and k:feedback=theta*(q[:,k-1]@lag.T)
                if theta and family=='A1-C':feedback=theta*q[:,k]
                raw=base+feedback;norm2=np.einsum('ni,ni->n',raw,raw)
                scale=np.minimum(1.,np.sqrt(np.maximum(0,remaining)*(1-1e-14)/np.maximum(norm2,1e-300)))
                a[:,k]=scale[:,None]*raw;clipped[:,k]=scale<1.
            spent=np.einsum('ni,ni->n',a[:,k],a[:,k])
            remaining=np.maximum(0,remaining-spent)
        used_budget[:,k]=executable_budget-remaining
    energy=np.einsum('ntm,ntm->n',a,a)
    assert energy.max()<=E
    assert not np.any(a[:,:Ta]) and not np.any(a[:,Ta+L:])
    return a,dict(energy=energy,clipped=clipped,used_budget=used_budget)

def old_sample_bank(model,indices):
    seeds=json.loads((P/'attack_gain_switch_audit_20261115/sampling_seed_map.json').read_text('utf8'))[model.name]
    xs=[];ys=[];qs=[];Ms=[];errors=[]
    for i in indices:
        path=ref.draw_physical_path(model,32,seeds['physical'][i])
        M,_=ref.original_watermarks(model,32,seeds['watermark'][i],.5,.5)
        q,received=ref.transmit_decode(model,path['y'],M)
        healthy=ref.run_common(model,path['y'],received)['posterior'][:,1]
        xs.append(path['x']);ys.append(path['y']);qs.append(q[:,0]);Ms.append(M[:,0]);errors.append(healthy-path['x'])
    return dict(x=np.array(xs),y=np.array(ys),q=np.array(qs),M=np.array(Ms),e=np.array(errors),
        indices=list(indices),physical_seeds=[seeds['physical'][i] for i in indices],
        watermark_seeds=[seeds['watermark'][i] for i in indices])
