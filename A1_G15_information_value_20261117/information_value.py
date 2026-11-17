"""All-waveform OL optimum and implementable sign policies, no physical sampling."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import sympy as s
from scipy.special import roots_laguerre
H=Path(__file__).resolve().parent;P=H.parent
SRC=P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json'
TA=7;L=10;T=32;E=10
def sm(x):return s.Matrix([[s.Rational(str(v)) for v in row] for row in x])
def fl(x):return np.array(x,float)
def coded_sign_policy(observed_q,unit_waveform,public_ell,E=10.):
    """Plan all supported actions from ONE legally observed coded packet.

    No realized secret, clean data, state, error, or future packet is accepted.
    Call with q6 for H or q7 for C at Ta7. Numerical interior protects the
    physical energy upper bound; the mathematical certificates use exact v.
    """
    q=np.asarray(observed_q,float)
    v=np.array([float(s.Rational(x)) for x in unit_waveform])
    ell=np.array([float(s.Rational(x)) for x in public_ell])
    sign=1. if ell@q>=0 else -1.
    word=np.sqrt(E)*(1-1e-12)*sign*v/np.linalg.norm(v)
    assert word@word<=E
    return word.reshape(-1,2)
def exact_model(name):
    raw=json.loads(SRC.read_text('utf8'));m=raw['models'][name];n=m['nodes'][1]
    A=sm(m['A']);K=sm(n['K']);G=sm(n['G']);Hs=[sm(x) for x in raw['H']];R=[sm(x) for x in raw['R']];Q=sm(raw['Q'])
    assert Hs[0]==s.eye(2) and R[0]==s.eye(2)/2
    J=s.eye(2)-K*Hs[1]-G*(Hs[0]+Hs[2]);F=J*A
    px=[];pc=[];ce=[];ey=[]
    noise=K*R[1]*K.T+G*(R[0]+R[2])*G.T
    for k in range(T):
        x=s.eye(2) if k==0 else A*px[-1]*A.T+Q
        c=-J*x if k==0 else F*ce[-1]*A.T-J*Q
        v=J*x*J.T+noise if k==0 else F*pc[-1]*F.T+J*Q*J.T+noise
        px.append(x);pc.append(v);ce.append(c);ey.append(c*Hs[0].T+G*R[0])
    powers=[s.eye(2)]
    for k in range(T):powers.append(powers[-1]*F)
    return dict(A=A,K=K,G=G,J=J,F=F,Hs=Hs,R=R,Q=Q,Px=px,Pe=pc,Ey=ey,powers=powers,weights={})
def cov_ey(m,k,t):
    assert k>=t
    return m['powers'][k-t]*m['Ey'][t]
def weights(m,t,ends):
    key=(t,tuple(ends))
    if key not in m['weights']:
        m['weights'][key]=sum((m['powers'][r-t].T*m['powers'][r-t] for r in ends if r>=t),s.zeros(2))
    return m['weights'][key]
def gram(m,times,ends):
    G=m['G'];F=m['F'];R=s.zeros(2*len(times))
    for i,t in enumerate(times):
        for j,u in enumerate(times):
            if t<=u:C=G.T*m['powers'][u-t].T*weights(m,u,ends)*G
            else:C=(G.T*m['powers'][t-u].T*weights(m,t,ends)*G).T
            if i==j:C=s.Rational(5,4)*C+s.trace(C)*s.eye(2)/4
            R[2*i:2*i+2,2*j:2*j+2]=C
    assert R==R.T
    return R
def unit_rational(vec):
    v=np.asarray(vec,float)
    if v[0]<0:v=-v
    ts=[s.Rational(round(float(z/(1+v[0]))*10**12),10**12) for z in v[1:]]
    p=sum(z*z for z in ts);u=s.Matrix([(1-p)/(1+p)]+[2*z/(1+p) for z in ts])
    assert (u.T*u)[0]==1
    return u
def coefficients(m,times,ends,v,obs):
    cs=[]
    for i,t in enumerate(times):
        C=m['G'].T*weights(m,t,ends)*cov_ey(m,t,obs)
        cs.append(v[2*i:2*i+2,0].T*C)
    return cs
def sign_cross(m,times,ends,v,obs,order):
    nr,na=order;r,w=roots_laguerre(nr);angles=(np.arange(na)+.5)*np.pi/na
    unit=np.stack([np.cos(angles),np.sin(angles)],1);outer=np.einsum('ni,nj->nij',unit,unit)
    M=.5*np.eye(2)+r[:,None,None,None]*outer[None]
    S=2*np.eye(2)-2*r[:,None,None,None]/(.5+r[:,None,None,None])*outer[None]
    cs=[fl(c).ravel() for c in coefficients(m,times,ends,v,obs)]
    ell=np.sum(cs,axis=0)
    # Publicly computed ell only; actual sign reads just ell.T q_obs.
    if obs<TA:
        numerator=np.einsum('i,rnij,j->rn',ell,S,ell)
    else:
        # Only first attack multiplier can be correlated with q_Ta.
        first=fl(v[:2,0]).ravel()
        C0=m['G'].T*weights(m,TA,ends)*cov_ey(m,TA,obs)
        row=np.einsum('i,rnij,jk->rnk',first,M,fl(C0))
        if len(cs)>1:row+=np.sum(cs[1:],axis=0)
        numerator=np.einsum('rni,rnij,j->rn',row,S,ell)
    Py=fl(m['Px'][obs]+m['R'][0]) # H0=I in this frozen instance
    z=np.einsum('rnij,j->rni',S,ell)
    variance=np.einsum('rni,ij,rnj->rn',z,Py,z)
    integral=np.sum(w[:,None]*(numerator/np.sqrt(variance)))/na*np.sqrt(2/np.pi)
    return float(integral),ell.tolist()
def history_genie_upper(m,k=7):
    # h_H=G.T E[e_k|q_history], fresh M mean I.
    C=s.Matrix.hstack(*[cov_ey(m,k,t) for t in range(k)])
    Y=s.zeros(2*k)
    for i in range(k):
        for j in range(k):
            if i>=j:V=m['A']**(i-j)*m['Px'][j]
            else:V=m['Px'][i]*(m['A']**(j-i)).T
            if i==j:V+=m['R'][0]
            Y[2*i:2*i+2,2*j:2*j+2]=V
    hcov=m['G'].T*C*Y.inv()*C.T*m['G']
    return hcov
def main():
    out={}
    for name in ('unstable','stable'):
        print('Constructing exact model '+name,flush=True);m=exact_model(name);z={}
        for label,times,ends in [('single',[7],[7]),('cumulative',list(range(7,17)),list(range(7,32)))]:
            R=gram(m,times,ends);vals,V=np.linalg.eigh(fl(R));v=unit_rational(V[:,-1])
            q=(v.T*R*v)[0];c=s.Matrix(sum(coefficients(m,times,ends,v,6),s.zeros(1,2))).T
            py=m['Px'][6]+m['R'][0]
            history=[];current=[]
            for order in [(32,64),(64,128),(96,256)]:
                x,ell=sign_cross(m,times,ends,v,6,order);history.append(dict(order=order,cross_unit=x))
                y,ellC=sign_cross(m,times,ends,v,7,order);current.append(dict(order=order,cross_unit=y))
            base=float(sum(s.trace(m['Pe'][r]) for r in ends))
            z[label]=dict(OL_optimum_increment=float(E*vals[-1]),OL_expected_MSE=base+float(E*vals[-1]),
              normal_MSE=base,near_optimal_waveform_unit=[str(a) for a in v],
              waveform_unit_energy_exact=True,OL_witness_increment=float(E*q),
              history_public_ell=[str(a) for a in c],current_public_ell=ellC,
              history_cross_quadrature=history,current_cross_quadrature=current,
              history_policy_increment=float(E*q)+2*np.sqrt(E)*history[-1]['cross_unit'],
              current_policy_increment=float(E*q)+2*np.sqrt(E)*current[-1]['cross_unit'],
              H_simple_strict_lower_cross=float(s.Rational(79,100)*abs(max(c,key=abs))/(3*s.sqrt(s.trace(py)))),
              moment_gram_shape=list(R.shape),quadrature_is_not_certified=True)
            if label=='single':
                hv=history_genie_upper(m)
                z[label]['H_genie_upper_increment']=float(E*vals[-1]+2*np.sqrt(E*float(s.trace(hv))))
                z[label]['H_genie_hcov_trace_exact']=str(s.trace(hv))
        out[name]=z
        print(json.dumps({name:{k:{x:r[x] for x in ['OL_optimum_increment','history_policy_increment','current_policy_increment']} for k,r in z.items()}}),flush=True)
    result=dict(date='2026-11-17',models=out,new_physical_samples=0,global_causal_optimum_computed=False,
      waveform_search='exact unrestricted OL eigenproblem, not a trained finite attack grid',
      canonical_model_SHA256=hashlib.sha256(json.dumps(json.loads(SRC.read_text('utf8')),sort_keys=True,separators=(',',':')).encode()).hexdigest())
    (H/'deterministic_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
