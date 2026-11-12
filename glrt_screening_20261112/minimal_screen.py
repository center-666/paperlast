"""Frozen-model deterministic screening; no physical/noise simulation.

All 1024 sign words are enumerated. Gaussian averages use fixed, unscrambled
Sobol quadrature, and are diagnostic only. Risk bounds do not assume independent
overlapping alarms. They are not exact first-alarm probabilities.
"""
from pathlib import Path
from itertools import product
import json
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, ncx2, qmc

HERE = Path(__file__).resolve().parent
P = HERE.parent
old = json.loads((P/'checks/水印收束_理论与映射核验_20261105.json').read_text(encoding='utf-8'))
F, g = np.array(old['F']), np.array(old['g'])
W = np.array(old['W10'])
H = np.array([[1., t/10] for t in range(10)])
Sigma = np.array([[float(__import__('fractions').Fraction(v)) for v in row]
                  for row in old['monitor']['elementary_covariance']])
ev, V = np.linalg.eigh(W)
Winvhalf = (V/np.sqrt(ev))@V.T
alpha = .05/13
threshold = {d: chi2.isf(alpha, d) for d in range(1, 15)}
J = np.zeros((14, 16))
for i in range(14):
    J[i, i:i+3] = [1,-2,1]
assert np.linalg.eigvalsh(Sigma).min() > 0

def build(mult):
    n = len(mult)
    A = np.zeros((n,16,2))
    A[:,3:13,:] = mult[:,:,None]*H
    design = np.einsum('ij,wja->wia',J,A)
    gp, gs = [], []
    for k in range(13):
        D = design[:,:k+2]
        gp.append(np.einsum('wmi,mn,wnj->wij',D,np.linalg.inv(Sigma[:k+2,:k+2]),D))
        D = design[:,k:k+2]
        gs.append(np.einsum('wmi,mn,wnj->wij',D,np.linalg.inv(Sigma[k:k+2,k:k+2]),D))
    B = np.zeros((n,4,2))
    hist=[]
    for k in range(13):
        B = np.einsum('ij,wja->wia',F,B)
        if k < 10:
            B += g[None,:,None]*mult[:,k,None,None]*H[k,None,:]
        hist.append(B.copy())
    return dict(mult=mult, G=np.stack(gp,axis=1), S=np.stack(gs,axis=1),
                B=np.stack(hist,axis=1), design=design)

def qvalues(model, xi, kind):
    G = model['S'] if kind=='sliding' else model['G']
    lam = np.maximum(0,np.einsum('i,wkij,j->wk',xi,G,xi))
    dfs = np.full(13,2) if kind=='sliding' else (np.arange(2,15) if kind=='omni' else np.r_[1,np.full(12,2)])
    return ncx2.sf(np.array([threshold[int(d)] for d in dfs]),dfs,lam)

def summarize(model,xi,kind):
    alarm=qvalues(model,xi,kind)
    b=np.einsum('wkti,i->wkt',model['B'],xi)
    harms=np.sum(b*b,axis=2)
    hit=harms>=.1
    eligible=hit.any(axis=1)
    first=np.argmax(hit,axis=1)
    cum=alarm.cumsum(axis=1)
    largest=np.maximum.accumulate(alarm,axis=1)
    rows=np.arange(len(first))
    lower=np.where(eligible,np.maximum(0,1-cum[rows,first]),0)
    upper=np.where(eligible,1-largest[rows,first],0)
    return dict(risk_lower=float(lower.mean()),risk_upper=float(upper.mean()),
                danger_probability=float(eligible.mean()),
                detection_by_H_bounds=[float(alarm.max(axis=1).mean()),float(np.minimum(1,alarm.sum(axis=1)).mean())],
                alarm_by_k_lower=largest.mean(axis=0).tolist(),
                alarm_by_k_upper=np.minimum(1,cum).mean(axis=0).tolist())

def terminal_search(model,kind):
    def f(theta):
        xi=np.sqrt(10)*Winvhalf@np.array([np.cos(theta),np.sin(theta)])
        return float(qvalues(model,xi,kind)[:,-1].mean())
    grid=np.linspace(0,np.pi,129,endpoint=False)
    vals=np.array([f(t) for t in grid])
    best=[]
    for idx in np.argsort(vals)[:3]:
        step=np.pi/129
        z=minimize_scalar(f,bounds=(grid[idx]-step,grid[idx]+step),method='bounded',options={'xatol':1e-10})
        best.append((float(z.fun),float(z.x%np.pi)))
    value,theta=min(best)
    xi=np.sqrt(10)*Winvhalf@np.array([np.cos(theta),np.sin(theta)])
    return dict(PD_final_single_test=value,theta=theta,xi=xi.tolist(),energy=float(xi@W@xi),
                objective='last predeclared test, encoded energy exactly10; not event detection minimum')

def risk_search(model,kind):
    # A screening grid, not a global minimax solve. Radial search is required.
    candidates=[]
    energies=[.25,1.,2.,4.,6.,8.,10.]
    for theta in np.linspace(0,np.pi,33,endpoint=False):
        u=Winvhalf@np.array([np.cos(theta),np.sin(theta)])
        for e in energies:
            xi=np.sqrt(e)*u
            if xi@W@xi>10.:
                xi*=1-1e-14
            s=summarize(model,xi,kind)
            candidates.append((s['risk_lower'],xi,s))
    # Also inspect damage-activation energies of the selected attack direction.
    lo, xi, s=max(candidates,key=lambda x:x[0])
    u=xi/np.sqrt(xi@W@xi)
    b=np.einsum('wkti,i->wkt',model['B'],u)
    power=np.sum(b*b,axis=2)
    crit=.1/np.maximum(power,1e-300)
    # Finite sign words permit exact finite critical radii, but checking all
    # directions/words is not needed for the predeclared minimum screening.
    # Use 17 ordered critical quantiles for this direction, reported as finite.
    c=crit[(crit>0)&(crit<=10)]
    if c.size:
        for e in np.unique(np.quantile(c,np.linspace(0,1,17))):
            xi=np.sqrt(e)*(1+1e-12)*u
            if xi@W@xi<=10.:
                s=summarize(model,xi,kind)
                candidates.append((s['risk_lower'],xi,s))
    lo, xi, s=max(candidates,key=lambda x:x[0])
    assert xi@W@xi <= 10.
    return dict(best_fixed_attack_lower=lo,xi=xi.tolist(),energy=float(xi@W@xi),
                fixed_attack_bounds=s,screen_points=len(candidates),
                not_a_global_upper_bound=True)

def sign_uniform_caps(model):
    harms=np.einsum('wkti,wktj->wkij',model['B'],model['B'])
    caps=np.empty((len(harms),13))
    for w in range(len(harms)):
        for k in range(13):
            caps[w,k]=10*eigh(harms[w,k],W,eigvals_only=True)[-1]
    possible=caps>=.1
    candidates=possible.any(axis=1)
    out={}
    for kind in ['sliding','glrt','omni']:
        word_bounds=np.zeros(len(harms))
        for w in np.flatnonzero(candidates):
            for k in np.flatnonzero(possible[w]):
                G=model['S' if kind=='sliding' else 'G'][w,k]
                if np.linalg.eigvalsh(G)[0] <= 1e-12:
                    gamma=0.
                else:
                    gamma=1/eigh(harms[w,k],G,eigvals_only=True)[-1]
                df=2 if kind=='sliding' else (k+2 if kind=='omni' else 2)
                bound=ncx2.cdf(threshold[df],df,.1*gamma)
                word_bounds[w]=max(word_bounds[w],float(bound))
        out[kind]=float(word_bounds.mean())
    return dict(max_possible_damage_by_k=caps.max(axis=0).tolist(),
                possibly_dangerous_words_by_k=possible.sum(axis=0).tolist(),
                possibly_dangerous_words=int(candidates.sum()),
                no_detector_uniform_risk_cap=float(candidates.mean()),
                detector_uniform_risk_caps=out,
                numerical_model_only=True,validated_interval_certificate=False), candidates

def main():
    signs=build(np.array(list(product([-1.,1.],repeat=10))))
    none=build(np.ones((1,10)))
    # Verify prefix designs and the two startup rows independently.
    assert np.allclose(signs['design'][:,0],0)
    assert np.allclose(np.linalg.det(signs['design'][:,1:3]), .1*signs['mult'][:,0]*signs['mult'][:,1])
    assert np.linalg.eigvalsh(signs['G'][:,1:]).min()>0
    assert np.allclose(none['design'][0,3:11],0)
    assert np.allclose(signs['B'][:,-1].mean(axis=0),0,atol=1e-14)
    uniform, candidate_words=sign_uniform_caps(signs)
    # Omit words proven incapable of any danger when optimizing risk, but keep
    # the original 1024 denominator by representing their contributions as zero.
    comparisons={}
    for name,model,kind in [('none_parity',none,'sliding'),('none_glrt',none,'glrt'),
                            ('none_prefix_omni',none,'omni'),
                            ('sign_parity',signs,'sliding'),('sign_glrt',signs,'glrt'),
                            ('sign_prefix_omni',signs,'omni')]:
        print('screen',name,flush=True)
        comparisons[name]=dict(terminal=terminal_search(model,kind),risk=risk_search(model,kind))
    # All Gaussian quadrature nodes are deterministic; no sampled physical
    # trajectories. Two resolutions measure stability only, not certified error.
    gauss=[]
    for power in [9,10]:
        nodes=qmc.Sobol(d=10,scramble=False).random_base2(power)
        mult=(1+chi2.ppf(nodes,1))/np.sqrt(2)
        model=build(mult)
        print('screen Gaussian',len(nodes),flush=True)
        r=dict(nodes=len(nodes),terminal=terminal_search(model,'glrt'),risk=risk_search(model,'glrt'))
        gauss.append(r)
    xi=np.array([1.,0.])
    mcrit=np.sqrt(.1/(g@g))
    # First-slot danger on [mcrit,2mcrit], no earlier attack-induced danger.
    Gfirst=float(np.linalg.inv(Sigma[:2,:2])[1,1])
    mass=chi2.cdf(np.sqrt(2)*2*mcrit-1,1)-chi2.cdf(np.sqrt(2)*mcrit-1,1)
    bound=mass*ncx2.cdf(threshold[1],1,(2*mcrit)**2*Gfirst)
    out=dict(date='2026-11-12',model='frozen four-dimensional single-link map',
             alpha_total_upper=.05,alpha_per_test=alpha,D=.1,energy_budget=10,
             prefixes=list(range(2,15)),prefix_rank=[1]+[2]*12,
             sign_uniform=uniform,comparisons=comparisons,
             gaussian_sobol_diagnostics=gauss,
             gaussian_first_slot_analytic_positive_bound=dict(xi=xi.tolist(),mcrit=float(mcrit),
                 event_multiplier_interval=[float(mcrit),float(2*mcrit)],probability_lower=float(bound),
                 expression='Pr(mcrit<=m<=2*mcrit)*CDF_ncx2(df1,threshold1,(2*mcrit)^2*Sigma2inv[1,1])',
                 proof_positive=True,numerical_evaluation_not_interval=True),
             physical_trajectories=0,random_draws=0,actual_first_alarm_distribution_computed=False,
             global_risk_optimized=False,GLRT_is_standard_baseline=True,
             gaussian_risk_bounds_are_quadrature_diagnostics_not_certificates=True)
    (HERE/'minimal_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(HERE/'frozen_sign_maps.npz',G=signs['G'],S=signs['S'],B=signs['B'],
                        Sigma=Sigma,W=W,mult=signs['mult'])
    print(json.dumps(dict(uniform=uniform,summary={n:r['risk']['best_fixed_attack_lower'] for n,r in comparisons.items()},
                         gaussian=[(r['nodes'],r['risk']['best_fixed_attack_lower']) for r in gauss]),indent=2))

if __name__=='__main__':
    main()
