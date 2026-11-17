"""Conditional trust-region optimizer and current-packet-only posterior probes.

No realized M, local measurement, state, error, or clean copy is an input.
Quadrature approximates a posterior; its optimum is NOT the exact Bayesian V_C.
"""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm,logsumexp
import information_value as iv
H=Path(__file__).resolve().parent
def maximize_quadratic(Q,h,R):
    d,U=np.linalg.eigh(Q);z=U.T@h;top=d[-1];tol=1e-13
    topids=np.abs(d-top)<tol
    a=np.zeros_like(z)
    if np.linalg.norm(z[topids])<tol:
        a[~topids]=z[~topids]/(top-d[~topids])
        if a@a<=R:
            a[np.where(topids)[0][-1]]=np.sqrt(max(0,R-a@a));nu=top
        else:
            nu=None
    else:nu=None
    if nu is None:
        # Solve in delta=nu-lambda_max to avoid loss of precision close to
        # the top eigenvalue, rather than subtracting nearly equal nu,d.
        gaps=top-d
        def f(delta):
            if delta==0:
                if np.linalg.norm(z[topids])>0:return np.inf
                return np.sum((z[~topids]/gaps[~topids])**2)-R
            return np.sum((z/(delta+gaps))**2)-R
        hi=max(1.,np.linalg.norm(h)/np.sqrt(R))
        while f(hi)>0:hi*=2
        delta=brentq(f,0.,hi,xtol=1e-18,rtol=1e-14)
        nu=top+delta;a=z/(delta+gaps)
    a=U@a
    return a,nu
def reduced_posterior(Qy,Cey,G,q,order):
    x,w=roots_hermitenorm(order);xx,yy=np.meshgrid(x,x,indexing='ij')
    beta=np.stack([xx.ravel(),yy.ravel()],1)/np.sqrt(2)
    M=.5*np.eye(2)+np.einsum('ni,nj->nij',beta,beta)
    y=np.einsum('nij,j->ni',M,q)
    loglike=np.log(np.linalg.det(M))-.5*np.einsum('ni,ij,nj->n',y,np.linalg.inv(Qy),y)
    logw=(np.log(w[:,None])+np.log(w[None,:])).ravel()+loglike
    p=np.exp(logw-logsumexp(logw))
    # E[e0|y]=Cey Py^-1 y is a PUBLIC Gaussian regression, not a true error sample.
    L=Cey@np.linalg.inv(Qy);mu=y@L.T
    Q=np.einsum('n,nij,jk,nkl->il',p,M,G.T@G,M)
    h=np.einsum('n,nij,jk,nk->i',p,M,G.T,mu)
    return Q,h
def main():
    rows={}
    for name in ('unstable','stable'):
        m=iv.exact_model(name);Py=iv.fl(m['Px'][7]+m['R'][0]);C=iv.fl(m['Ey'][7]);G=iv.fl(m['G'])
        out=[]
        for q in ([0.,0.],[1.,0.],[0.,1.],[1.,1.]):
            probes=[]
            for order in (24,40,64):
                Q,h=reduced_posterior(Py,C,G,np.array(q),order);a,nu=maximize_quadratic(Q,h,10*(1-1e-12))
                # Sufficient GLOBAL condition for this APPROXIMATE conditional Q/h.
                residual=float(np.linalg.norm((nu*np.eye(2)-Q)@a-h))
                norm=float(a@a)
                assert residual<1e-9 and abs(norm-10)<1e-8 and norm<=10
                probes.append(dict(order=order,Q=Q.tolist(),h=h.tolist(),attack=a.tolist(),
                  objective_increment=float(a@Q@a+2*h@a),nu=float(nu),
                  stationarity_residual=residual,energy=norm,
                  minimum_dual_eigenvalue=float(np.linalg.eigvalsh(nu*np.eye(2)-Q).min())))
            out.append(dict(observed_q=q,probes=probes))
        rows[name]=out
    result=dict(date='2026-11-17',models=rows,
      information_used='only current target coded q and public model; ignores available history',
      numerical_budget_reserve='1e-12 relative interior for probes; physical bound stays10',
      posterior_quadrature_certified=False,full_C_Bayesian_optimum_computed=False,
      point_probes_have_no_unconditional_performance_claim=True,new_physical_samples=0)
    (H/'conditional_posterior_probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(dict(probes_per_model=4,orders=[24,40,64],posterior_not_certified=True)))
if __name__=='__main__':main()
