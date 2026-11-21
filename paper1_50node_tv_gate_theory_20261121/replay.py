"""Algebraically equivalent 50-node raw-measurement replay with isolated RNG.

The information-form 2x2 calculation is exact algebraically; source validation
checks finite-precision equivalence. All 50 physical sensor paths are used.
Only the five affected receivers are recomputed under a gate-policy change;
the other 45 provably remain equal to their base no-gate paths.
"""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import pinv
from source_model import inv2,matvec


def base_filter(model,physical,schedule):
    B=len(physical.x);post=np.zeros((B,150,50,2))
    for k in range(1,150):
        post[:,k]=matvec(schedule.F[k][None],post[:,k-1])+matvec(schedule.star[k][None],physical.suff[:,k])
    return post


def scores(model,physical,delta):
    """Same raw packet windows, residual buffers, covariance and KL as source."""
    B=len(physical.x);E=len(model.links);out=np.zeros((B,150,E))
    packets=physical.y[:,:,model.links[:,1]].copy()+delta
    packets[:,0]=0.  # Source communicates from k=1; received history at k=0 is zero.
    inverses=np.stack([pinv(t) for t in model.Theta]);logtheta=np.linalg.slogdet(model.Theta)[1]
    history=[]
    for k in range(4,150):
        raw=packets[:,k-4:k+1].transpose(0,2,1,3).reshape(B,E,10)
        residual=np.einsum('edm,bem->bed',model.Z,raw)
        history.append(residual)
        if len(history)>4:history.pop(0)
        if len(history)<2:continue
        R=np.stack(history,axis=-1);mean=R.mean(axis=-1);centered=R-mean[:,:,:,None]
        covariance=np.einsum('bedt,beft->bedf',centered,centered)/(len(history)-1)
        covariance+=1e-6*np.eye(8)
        trace=np.einsum('eij,beji->be',inverses,covariance)
        quad=np.einsum('bei,eij,bej->be',mean,inverses,mean)
        signs,logcov=np.linalg.slogdet(covariance)
        value=.5*(trace+quad-8+logtheta[None]-logcov)
        out[:,k]=np.where(signs>0,np.maximum(0.,value),0.)
    return out


def affected_filter(model,physical,delta,alarms,threshold,gate=True,gate_override=None):
    B=len(physical.x);ids=model.receiver_ids;L=len(ids)
    post=np.zeros((B,150,L,2));P=np.broadcast_to(5*np.eye(2),(B,L,2,2)).copy()
    covariance=np.zeros((B,150,L,2,2));covariance[:,0]=P
    stars=np.zeros_like(covariance);Fs=np.zeros_like(covariance)
    active_gates=(alarms>threshold) if gate else np.zeros_like(alarms,dtype=bool)
    if gate_override is not None:active_gates=np.asarray(gate_override,dtype=bool)
    for k in range(1,150):
        info=np.broadcast_to(model.Jfull[ids],(B,L,2)).copy();suff=physical.suff[:,k,ids].copy()
        for e,(receiver,sender) in enumerate(model.links):
            i=model.receiver_index[int(receiver)];g=active_gates[:,k,e]
            contribution=2*delta[:,k,e]*model.hdiag[sender]
            suff[:,i]+=(~g)[:,None]*contribution-g[:,None]*physical.weighted[:,k,sender]
            info[:,i]-=g[:,None]*model.info_packet[sender]
        pred=model.A@P@model.A.T+model.Q;precision=inv2(pred)
        total=precision.copy();total[...,0,0]+=info[...,0];total[...,1,1]+=info[...,1]
        star=inv2(total);F=star@precision@model.A
        post[:,k]=matvec(F,post[:,k-1])+matvec(star,suff)
        P=star+1e-9*np.eye(2)
        covariance[:,k]=P;stars[:,k]=star;Fs[:,k]=F
    return SimpleNamespace(post=post,P=covariance,star=stars,F=Fs,gates=active_gates,scores=alarms)


def attack_watermarks(model,count,theta,radial_seed,initial_seed,direction_seed):
    """Stationary attack-window chains; identical first M and spectra across laws."""
    E=len(model.links);m=2;length=60
    radii=.5*np.random.default_rng(radial_seed).chisquare(m,size=(count,length,E))
    u=np.random.default_rng(initial_seed).normal(size=(count,E,m));u/=np.linalg.norm(u,axis=-1,keepdims=True)
    rng=np.random.default_rng(direction_seed);beta=np.empty((count,length,E,m))
    for k in range(length):
        if k:
            old=u.reshape(-1,m);new=rng.normal(size=old.shape);new/=np.linalg.norm(new,axis=1,keepdims=True)
            pending=np.arange(len(old))
            while len(pending):
                t=np.sum(new[pending]*old[pending],axis=1)**2
                accept=rng.random(len(pending))<np.exp(theta*t-max(theta,0.))
                old[pending[accept]]=new[pending[accept]];pending=pending[~accept]
                if len(pending):
                    proposal=rng.normal(size=(len(pending),m));proposal/=np.linalg.norm(proposal,axis=1,keepdims=True)
                    new[pending]=proposal
            u=old.reshape(count,E,m)
        beta[:,k]=np.sqrt(radii[:,k,:,None])*u
    attack=model.attack[50:110]
    dot=np.einsum('btem,tem->bte',beta,attack)
    decoded=.5*attack[None]+beta*dot[...,None]
    delta=np.zeros((count,150,E,m));delta[:,50:110]=decoded
    return delta,beta


def paired_forcing(model,physical,healthy,attacked,delta):
    """Verify the exact endogenous-gate bias identity, without discarding cross terms."""
    B=len(physical.x);ids=model.receiver_ids;L=len(ids)
    b=attacked.post-healthy.post
    mismatch=np.zeros((B,150,L,2));forcing=np.zeros_like(mismatch)
    residual=np.zeros_like(mismatch)
    pred0=healthy.post[:,:-1]@model.A.T
    base=physical.suff[:,1:,ids]-model.Jfull[ids][None,None]*pred0
    Sa=base.copy();S0=base.copy();D=np.zeros_like(base)
    for e,(receiver,sender) in enumerate(model.links):
        i=model.receiver_index[int(receiver)];ga=attacked.gates[:,1:,e];g0=healthy.gates[:,1:,e]
        innovation=physical.weighted[:,1:,sender]-model.info_packet[sender]*pred0[:,:,i]
        Sa[:,:,i]-=ga[:,:,None]*innovation;S0[:,:,i]-=g0[:,:,None]*innovation
        D[:,:,i]+=(~ga)[:,:,None]*(2*model.hdiag[sender]*delta[:,1:,e])
    mismatch[:,1:]=matvec(attacked.star[:,1:],Sa)-matvec(healthy.star[:,1:],S0)
    forcing[:,1:]=matvec(attacked.star[:,1:],D)
    residual[:,1:]=b[:,1:]-matvec(attacked.F[:,1:],b[:,:-1])-mismatch[:,1:]-forcing[:,1:]
    return SimpleNamespace(b=b,mismatch=mismatch,forcing=forcing,residual=residual)
