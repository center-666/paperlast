"""Deterministic time-varying no-gate moments and raw-buffer support analysis."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
from scipy.linalg import block_diag,null_space

HERE=Path(__file__).resolve().parent
prior=HERE.parent/'paper1_SSCD_review_20261121'
sys.path.insert(0,str(prior))
import sscd


def polynomial(coeff,z):
    out=np.zeros_like(coeff[0])
    for c in reversed(coeff):out=out*z+c
    return (out+out.T)/2


def source_templates(model):
    T=np.zeros((150,6,2,2));power=np.eye(2)
    for t in range(60):
        for e,(i,j) in enumerate(model.links):T[50+t,e]=model.H[j]@power
        power=model.A@power
    E=np.einsum('teia,teib->ab',T,T)
    return T,E


def state_polynomials(model,schedule):
    """Cost-to-go contraction; common attack parameter, six independent WM sources."""
    T,E=source_templates(model);mu=1.;q=.5;m=2;weight=1/(50*150)
    coeff=np.zeros((60,2,2));meanQ=np.zeros((2,2));radial=np.zeros((2,2));B=np.zeros((150,6,2,2))
    for k in range(1,150):
        for e,(i,j) in enumerate(model.links):B[k,e]=schedule.star[k,i]@(2*model.H[j].T)
    for receiver in model.receiver_ids:
        i=int(receiver);edges=np.flatnonzero(model.links[:,0]==i)
        S=np.zeros((150,2,2));S[-1]=weight*np.eye(2)
        for k in range(148,-1,-1):S[k]=weight*np.eye(2)+schedule.F[k+1,i].T@S[k+1]@schedule.F[k+1,i]
        mean=np.zeros((2,2))
        for k in range(1,150):
            mean=schedule.F[k,i]@mean+sum((B[k,e]@T[k,e] for e in edges),np.zeros((2,2)))
            meanQ+=mu*mu*weight*mean.T@mean
        for e in edges:
            for ell in range(50,110):
                P=B[ell,e].T@S[ell]@B[ell,e]
                diag=T[ell,e].T@(np.trace(P)*np.eye(2)+P)@T[ell,e]
                coeff[0]+=q*q*diag
                radial+=2*q*q/(m+2)*T[ell,e].T@(np.trace(P)*np.eye(2)+2*P)@T[ell,e]
            for r in range(51,110):
                phi=np.eye(2)
                for ell in range(r-1,49,-1):
                    phi=phi@schedule.F[ell+1,i]
                    A=B[r,e].T@S[r]@phi@B[ell,e]
                    cross=T[ell,e].T@(np.trace(A)*np.eye(2)+A-(2/m)*A.T)@T[r,e]
                    coeff[r-ell]+=q*q*m/(m+2)*(cross+cross.T)
    coeff[0]+=meanQ
    return dict(coeff=coeff,E=E,meanQ=meanQ,radial_floor=radial,templates=T,gain_blocks=B)


def buffer_support(model,e):
    """Four overlapping 8-D residuals from eight 2-D raw packet samples."""
    sender=int(model.links[e,1]);H=model.H[sender];L=8
    P=np.zeros((32,16))
    for j in range(4):P[8*j:8*(j+1),2*j:2*j+10]=model.Z[e]
    O=np.vstack([H@np.linalg.matrix_power(model.A,t) for t in range(L)])
    Hw=np.zeros((L*2,(L-1)*2))
    for r in range(L):
        for t in range(r):Hw[2*r:2*r+2,2*t:2*t+2]=H@np.linalg.matrix_power(model.A,r-1-t)
    V=Hw@np.kron(np.eye(L-1),model.Q)@Hw.T+np.kron(np.eye(L),model.R[sender])
    U,s,_=np.linalg.svd(P,full_matrices=False);rank=int(np.sum(s>1e-11))
    U=U[:,:rank];Z=U.T@P;theta=Z@V@Z.T
    assert rank==14 and np.linalg.norm(P@O)<1e-10
    np.linalg.cholesky(theta)
    T=[H@np.linalg.matrix_power(model.A,k) for k in range(L)]
    whiten=np.linalg.solve(np.linalg.cholesky(theta),Z)
    q=.5;m=2;G=np.zeros((L,2,2))
    for ell in range(L):
        zl=whiten[:,2*ell:2*ell+2];W=zl.T@zl
        G[0]+=q*q*T[ell].T@(np.trace(W)*np.eye(m)+W)@T[ell]
        for r in range(ell+1,L):
            zr=whiten[:,2*r:2*r+2];A=zr.T@zl
            cross=T[ell].T@(np.trace(A)*np.eye(m)+A-(2/m)*A.T)@T[r]
            G[r-ell]+=q*q*m/(m+2)*(cross+cross.T)
    return dict(sender=sender,P=P,U=U,Z=Z,theta=theta,trace_coeff=G,templates=T,rank=rank,
                raw_packet_dimension=16,unprojected_residual_dimension=32)


def dg_direction(buffer,xi,zeta):
    d=np.stack([t@xi for t in buffer['templates']])
    cov=sscd.full_watermark_covariance(d,zeta,.5)
    return sscd.gaussian_covariance_kl_proxy(buffer['theta'],buffer['Z']@cov@buffer['Z'].T)


def design(model,state,buffers):
    E=state['E'];ev,V=np.linalg.eigh(E);Einv=(V/np.sqrt(ev))@V.T
    energy=float(model.xi@E@model.xi)
    domain=sscd.dependence_interval(2,8,.7)
    grid=np.unique(np.r_[np.linspace(*domain['zeta_interval'],321),0.])
    Q0=state['coeff'][0];rows=[]
    for z in grid:
        changes=energy*float(np.linalg.eigvalsh(Einv@(polynomial(state['coeff'],z)-Q0)@Einv)[-1])
        lower=[]
        for bf in buffers:
            G=polynomial(bf['trace_coeff'],z);t=max(0.,energy*np.linalg.eigvalsh(Einv@G@Einv)[0]);d=bf['rank']
            lower.append(float(.5*d*(t/d-np.log1p(t/d))))
        theta=sscd.theta_from_zeta(2,float(z))
        rows.append(dict(zeta=float(z),theta=theta,robust_jensen_lower=min(lower),
                         per_link_jensen_lower=lower,state_directionwise_change=changes,
                         feasible=bool(changes<=1e-9),info_per_step=sscd.watson_stats(2,theta)['mi_nats']))
    iid=next(r for r in rows if r['zeta']==0.)
    best=max((r for r in rows if r['feasible']),key=lambda r:r['robust_jensen_lower'])
    if best['robust_jensen_lower']<=iid['robust_jensen_lower']+1e-10:best=iid
    # Any evaluated direction is an UPPER bound for the true worst DG.
    upper=[]
    for bf in buffers:
        G=bf['trace_coeff'][0];_,vec=np.linalg.eigh(Einv@G@Einv)
        xi=np.sqrt(energy)*Einv@vec[:,0];upper.append(dg_direction(bf,xi,0.))
    iid_upper=min(upper)
    witness=best['robust_jensen_lower']>iid_upper
    conservative=best if witness else iid
    summary=[]
    for theta in (0.,-1.315236976827179,1.315236976827179,best['theta']):
        z=sscd.watson_stats(2,theta)['zeta'];Q=polynomial(state['coeff'],z)
        summary.append(dict(theta=theta,zeta=z,original_direction_risk=float(model.xi@Q@model.xi),
                             worst_fixed_energy_risk=float(energy*np.linalg.eigvalsh(Einv@Q@Einv)[-1])))
    raw_iid=float(model.xi@Q0@model.xi);floor=float(model.xi@(state['meanQ']+state['radial_floor'])@model.xi)
    radius=max(abs(z) for z in domain['zeta_interval'])
    family_bound=sum(radius**h*abs(float(model.xi@state['coeff'][h]@model.xi)) for h in range(1,60))
    return dict(scope='321-point signed design of a valid Jensen lower bound; not a global optimizer or actual detector-power objective',
        energy_original_waveform=energy,energy_from_exact_source_sequence=model.coded_energy,
        energy_gram=E.tolist(),domain=domain,iid=iid,bound_optimum=best,
        iid_worst_dg_upper_witness=iid_upper,strict_worst_dg_gain_witness=bool(witness),
        conservative_proxy_policy=conservative,
        original_direction_mean_floor=float(model.xi@state['meanQ']@model.xi),
        original_direction_mean_plus_radial_floor=floor,
        original_direction_iid_risk=raw_iid,max_possible_relative_raw_risk_reduction=1-floor/raw_iid,
        signed_family_absolute_change_bound=family_bound,
        signed_family_relative_change_bound=family_bound/raw_iid,
        scheme_risks=summary,grid=rows)
