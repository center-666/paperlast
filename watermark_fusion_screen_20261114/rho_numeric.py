"""Deterministic, bounded-size normal-MSE and first-packet risk screen.

All watermark-integral and optimization numbers are diagnostics, not certified
worst event risks. No physical/noise Monte Carlo or original scripts called.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import brentq,minimize_scalar
from scipy.stats import ncx2,chi2
H=Path(__file__).resolve().parent
MODEL=H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
data=json.loads(MODEL.read_text(encoding='utf-8'))
scope=json.loads((H/'frozen_scope.json').read_text(encoding='utf-8'))
ETAS=scope['normal_MSE_relative_budgets'];E=10.;D=.1
TAU=float(chi2.isf(.05/15,8))

def normGM(G,u,angles):
    v=np.stack([np.cos(angles),np.sin(angles)],axis=-1)
    T=G.T@G
    t=np.trace(T)+(np.square(1+2*u)-1)*np.einsum('...i,ij,...j->...',v,T,v)
    det=np.linalg.det(G)**2*np.square(1+2*u)
    return np.sqrt((t+np.sqrt(np.maximum(t*t-4*det,0)))/2)

def rho_moments(G,gamma,nrad=64,nang=64):
    angles=np.pi*(np.arange(nang)+.5)/nang
    x,w=roots_laguerre(nrad)
    base=np.linalg.norm(G,2)
    if gamma==0:return (0.,0.)
    cut=[]
    for angle in angles:
        if gamma<=base:cut.append(0.);continue
        hi=max(1.,gamma/base)
        while normGM(G,hi,angle)<gamma:hi*=2
        cut.append(brentq(lambda u:float(normGM(G,u,angle)-gamma),0,hi,xtol=1e-13))
    cut=np.array(cut)
    den=normGM(G,cut[:,None]+x[None,:],angles[:,None])
    rho=np.minimum(1.,gamma/den)
    tail=np.exp(-cut)
    return tuple(float(np.mean(1-tail+tail*(rho**p@w))) for p in [1,2])

def model_params(name):
    m=data['models'][name];node=m['nodes'][1]
    A=np.array(m['A']);G=np.array(node['G']);K=np.array(node['K'])
    Hi=np.array(data['H'][1]);Hj=np.array(data['H'][0]);Ho=np.array(data['H'][2])
    Q=np.eye(2);R=.5*np.eye(2)
    J0=np.eye(2)-K@Hi-G@Ho;Y=G@Hj
    F0=J0@A;X=Y@A
    return dict(A=A,G=G,K=K,Hi=Hi,Hj=Hj,Ho=Ho,Q=Q,R=R,J0=J0,Y=Y,F0=F0,X=X,
                reference_trace=float(np.trace(node['posterior_covariance'])))

def stationary(p,m1,m2):
    J0,Y,F0,X=p['J0'],p['Y'],p['F0'],p['X']
    Q,R,K,G=p['Q'],p['R'],p['K'],p['G']
    op=np.kron(F0,F0)-m1*(np.kron(F0,X)+np.kron(X,F0))+m2*np.kron(X,X)
    noise=J0@Q@J0.T-m1*(J0@Q@Y.T+Y@Q@J0.T)+m2*Y@Q@Y.T+K@R@K.T+G@R@G.T+m2*G@R@G.T
    P=np.linalg.solve(np.eye(4)-op,noise.reshape(-1)).reshape(2,2)
    assert np.max(abs(P-P.T))<1e-12 and np.linalg.eigvalsh(P).min()>0
    assert max(abs(np.linalg.eigvals(op)))<1
    return float(np.trace(P)),P

def choose_params(p,eta,nrad=64,nang=64):
    ref=p['reference_trace'];target=(1+eta)*ref
    f=lambda r:stationary(p,r,r*r)[0]-target
    assert f(0)>0 and f(1)<0
    r=brentq(f,0,1,xtol=1e-12)
    def fa(g):
        mm=rho_moments(p['G'],g,nrad,nang)
        return stationary(p,*mm)[0]-target
    hi=np.linalg.norm(p['G'],2)
    while fa(hi)>0:hi*=2
    gamma=brentq(fa,0,hi,xtol=1e-11)
    m1,m2=rho_moments(p['G'],gamma,nrad,nang)
    return dict(eta=eta,fixed_rho=r,gamma=gamma,aware_Erho=m1,aware_Erho2=m2,
        fixed_trace=stationary(p,r,r*r)[0],aware_trace=stationary(p,m1,m2)[0],
        target_trace=target)

def first_packet_metric(A):
    s=4;Q=np.eye(2);R=.5*np.eye(2);T=np.zeros((8,8))
    for j in range(s):
        T[2*j:2*j+2,2*j:2*j+2]=Q+R+A@R@A.T
        if j+1<s:
            T[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T
            T[2*j+2:2*j+4,2*j:2*j+2]=-A@R
    return np.linalg.inv(T)[-2:,-2:]

def fixed_sign_risk(G,C,rho,theta):
    u=np.array([np.cos(theta),np.sin(theta)])
    gain=float(np.linalg.norm(rho*G@u))
    r=np.sqrt(D)/gain
    if r>np.sqrt(E):return (0.,None)
    # at >=D convention, risk decreases beyond the minimum dangerous radius.
    lam=float(r*r*(u@C@u))
    return (float(ncx2.cdf(TAU,8,lam)),r)

def maximize_fixed_sign(G,C,rho):
    ts=np.linspace(0,np.pi,257)
    vals=[fixed_sign_risk(G,C,rho,t)[0]for t in ts]
    i=int(np.argmax(vals));lo=ts[max(0,i-1)];hi=ts[min(len(ts)-1,i+1)]
    opt=minimize_scalar(lambda t:-fixed_sign_risk(G,C,rho,t)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-13})
    risk,r=fixed_sign_risk(G,C,rho,float(opt.x))
    return dict(risk=risk,direction_angle=float(opt.x),radius=r,method='direction grid + local refine; analytic radial boundary',global_certified=False)

def main():
    out=dict(date='2026-11-14',kind='stationary moment solver and deterministic first-packet diagnostic',
        model_sha256=hashlib.sha256(MODEL.read_text(encoding='utf-8-sig').encode()).hexdigest(),
        E=E,D=D,normal_budgets=ETAS,detector_tau=TAU,detector_df=8,
        alpha_test=.05/15,total_false_alarm_union_bound=.05,
        physical_noise_samples=0,full_event_worst_risk_certified=False,models={})
    for name in data['models']:
        p=model_params(name);C=first_packet_metric(p['A'])
        ref,P=stationary(p,1,1);assert abs(ref-p['reference_trace'])<1e-12
        rows=[choose_params(p,e)for e in ETAS]
        for row in rows:
            row['fixed_sign_first_packet']=maximize_fixed_sign(p['G'],C,row['fixed_rho'])
            chk=choose_params(p,row['eta'],128,128)
            row['quadrature_64_to_128_gamma_difference']=chk['gamma']-row['gamma']
            row['high_order']=chk
        out['models'][name]=dict(reference_trace=ref,no_neighbor_rho0_trace=stationary(p,0,0)[0],
            no_neighbor_relative_loss=stationary(p,0,0)[0]/ref-1,
            F0_norm=float(np.linalg.norm(p['F0'],2)),F1_norm=float(np.linalg.norm(p['F0']-p['X'],2)),
            original_sign_first_packet=maximize_fixed_sign(p['G'],C,1),rows=rows)
        print(name,json.dumps(out['models'][name],ensure_ascii=False,indent=2))
    (H/'rho_normal_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
