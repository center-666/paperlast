"""First-packet Gaussian watermark risk diagnostic with split radial integration.

Danger boundaries are quadratic/quartic roots in ||beta||^2, not sampled
indicator functions. Outer angle/quadrature and direction search are still
floating diagnostics, NOT certified worst risks or unknown-start event risks.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import roots_laguerre,roots_legendre
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import ncx2
from rho_numeric import H,model_params,first_packet_metric,TAU,E,D

def positive_roots(p):
    p=np.array(p,float)
    while len(p)>1 and abs(p[-1])<1e-14*np.max(abs(p)):p=p[:-1]
    if len(p)<2:return []
    r=np.polynomial.polynomial.polyroots(p)
    return sorted(float(x.real)for x in r if abs(x.imag)<1e-7*(1+abs(x.real)) and x.real>1e-10)

def moments_at(G,C,w,v):
    B=G.T@G;vw=float(v@w);d=float(v@B@v)
    aa=np.array([w@B@w,2*vw*(v@B@w),vw*vw*d])
    ll=np.array([w@C@w,2*vw*(v@C@w),vw*vw*(v@C@v)])
    trace=np.array([np.trace(B),2*d,d])
    det=np.linalg.det(B)*np.array([1.,2.,1.])
    return aa,ll,trace,det

def normsq(trace,det,x):
    t=np.polynomial.polynomial.polyval(x,trace);dd=np.polynomial.polynomial.polyval(x,det)
    return (t+np.sqrt(np.maximum(t*t-4*dd,0)))/2

def integrate_risk(G,C,theta,energy,kind,param=1.,nang=32,nrad=24,adaptive=False,angular_split=False):
    if energy<=0:return 0.
    w=np.array([np.cos(theta),np.sin(theta)])
    angles=np.pi*(np.arange(nang)+.5)/nang
    angle_weights=np.full(nang,1/nang)
    if angular_split:
        # Resolve narrow cones at (v.w)(v.G'G w)=0 and far-tail cap crossings.
        bw=G.T@G@w
        critical=[0.,np.pi,(theta+np.pi/2)%np.pi,(np.arctan2(bw[1],bw[0])+np.pi/2)%np.pi]
        ratio=np.sqrt(D/(param*param*energy)) if kind=='aware' else 2.
        if ratio<1:
            h=np.arccos(ratio);critical += [(theta-h)%np.pi,(theta+h)%np.pi]
        critical=sorted(set(critical));tt,ww=roots_legendre(max(24,nang//(len(critical)-1)))
        aa=[];vv=[]
        for a,b in zip(critical,critical[1:]):
            aa.extend(a+(tt+1)*(b-a)/2);vv.extend(ww*(b-a)/(2*np.pi))
        angles=np.array(aa);angle_weights=np.array(vv)
    lg,lw=roots_laguerre(nrad);gx,gw=roots_legendre(nrad)
    total=0.
    for angle,angle_weight in zip(angles,angle_weights):
        v=np.array([np.cos(angle),np.sin(angle)])
        aa,ll,tr,det=moments_at(G,C,w,v)
        roots=[]
        if kind=='fixed':
            p=energy*param**2*aa;p[0]-=D;roots+=positive_roots(p)
            def dangerous(x):return energy*param**2*np.polynomial.polynomial.polyval(x,aa)>=D
        elif kind=='aware':
            q=energy*param**2/D*aa
            # n(x)<=q(x); det(q I - M B M)=0 locates potential boundaries.
            pp=np.polynomial.polynomial.polysub(np.polynomial.polynomial.polymul(q,q),np.polynomial.polynomial.polymul(tr,q))
            pp=np.polynomial.polynomial.polyadd(pp,det)
            roots+=positive_roots(pp)
            p=energy*aa;p[0]-=D;roots+=positive_roots(p)
            p=tr.copy(); # ||GM||=gamma cap transition, polynomial eigen condition.
            pp=np.polynomial.polynomial.polyadd(-param**2*tr,det);pp[0]+=param**4
            roots+=positive_roots(pp)
            def dangerous(x):
                n=normsq(tr,det,x)
                return energy*np.polynomial.polynomial.polyval(x,aa)*min(1.,param**2/n)>=D
        else:raise ValueError(kind)
        bounds=sorted(set([0.]+roots))+[np.inf]
        for lo,hi in zip(bounds,bounds[1:]):
            middle=(lo+hi)/2 if np.isfinite(hi) else lo+max(1.,lo*.1)
            if not dangerous(middle):continue
            a=lo/2
            if adaptive:
                def integrand(u):
                    lam=energy*np.polynomial.polynomial.polyval(2*u,ll)
                    return np.exp(-u)*ncx2.cdf(TAU,8,max(lam,0))
                total+=angle_weight*quad(integrand,a,hi/2,epsabs=2e-9,epsrel=2e-8,limit=160)[0]
                continue
            if np.isinf(hi):
                u=a+lg;weights=np.exp(-a)*lw
            else:
                b=hi/2;u=a+(gx+1)*(b-a)/2
                weights=gw*(b-a)/2*np.exp(-u)
            lam=energy*np.polynomial.polynomial.polyval(2*u,ll)
            total+=angle_weight*float(weights@ncx2.cdf(TAU,8,np.maximum(lam,0)))
    return total

def maximize_diagnostic(G,C,kind,param):
    # Declared bounded coarse grid, then local refine; no global certificate.
    ts=np.linspace(0,np.pi,13,endpoint=False);es=np.linspace(.05,E,21)
    vals=[]
    for t in ts:
        for e in es:vals.append((integrate_risk(G,C,t,e,kind,param,16,16),t,e))
    seeds=sorted(vals,reverse=True)[:2];found=[]
    for _,t,e in seeds:
        opt=minimize(lambda x:-integrate_risk(G,C,float(x[0]%np.pi),float(np.clip(x[1],0,E)),kind,param,32,24),
                     [t,e],method='Nelder-Mead',options={'maxiter':65,'xatol':2e-4,'fatol':3e-6})
        tt=float(opt.x[0]%np.pi);ee=float(np.clip(opt.x[1],0,E))
        risk=integrate_risk(G,C,tt,ee,kind,param,64,48)
        risk2=integrate_risk(G,C,tt,ee,kind,param,128,80)
        found.append(dict(risk=risk2,direction_angle=tt,energy=ee,
            quadrature_64_to_128_difference=risk2-risk,
            attack=(np.sqrt(ee)*np.array([np.cos(tt),np.sin(tt)])).tolist(),
            local_search_success=bool(opt.success)))
    best=max(found,key=lambda x:x['risk'])
    best.update(global_worst_certified=False,criterion='first posterior bias >=D and current ordinary parity does not alarm',
                quadrature_method='radial danger intervals split by quadratic/quartic roots; Gaussian CDF weighted by radial/angle quadrature',
                search='13 directions x 21 energies, two local refinements; no global claim')
    return best

def main():
    source=H/'rho_normal_screen.json';normal=json.loads(source.read_text(encoding='utf-8'))
    out=dict(date='2026-11-14',kind='finite first-packet diagnostic, not full-event worst risk',
        primary_attack='L=1 autonomous input a=Hj xi, actual energy <=10, no current secret knowledge',
        reference='same-noise same-secret ungated nominal shadow; current detector before fusion',
        detector='ordinary df8, alpha=.05/15, total 15-test FA <=.05 by union bound',
        larger_event_worst_risk_known=False,physical_noise_samples=0,models={})
    for name,m in normal['models'].items():
        p=model_params(name);C=first_packet_metric(p['A']);rows=[]
        original=maximize_diagnostic(p['G'],C,'fixed',1.)
        print(name,'original Gaussian',original,flush=True)
        out['models'][name]={'original_Gaussian':original,'original_iid_sign':m['original_sign_first_packet'],'rows':rows}
        for r in m['rows']:
            fixed=maximize_diagnostic(p['G'],C,'fixed',r['fixed_rho'])
            aware=maximize_diagnostic(p['G'],C,'aware',r['high_order']['gamma'])
            row=dict(eta=r['eta'],fixed_rho=r['fixed_rho'],gamma=r['high_order']['gamma'],
                no_watermark_fixed_rho=r['fixed_sign_first_packet'],Gaussian_fixed_rho=fixed,
                Gaussian_aware=aware,iid_sign_aware=r['fixed_sign_first_packet'])
            rows.append(row);print(name,'eta',r['eta'],'fixed',fixed['risk'],'aware',aware['risk'],
                                   'signfixed',r['fixed_sign_first_packet']['risk'],flush=True)
            (H/'rho_first_packet_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (H/'rho_first_packet_screen.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
