"""Fixed-estimator baseline powers and P1/P3 responses, no new watermark design.

Only the first complete attacked five-packet window is scored here. The
statistics are true conditional noncentral-chi-square tails averaged over
watermark paths, not a tail at the mean noncentrality. Searches and QMC values
are numerical diagnostics, not interval/global certificates.
"""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import chi2,ncx2,qmc,norm

H=Path(__file__).resolve().parent
MODEL=H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
data=json.loads(MODEL.read_text(encoding='utf-8'))
E=10.;D=.1;L=10;alpha=.05/15;TAU=float(chi2.isf(alpha,8))
v,ve=quad(lambda u:np.exp(-u)/(1+2*u)**2,0,np.inf,epsabs=1e-12)
NU=.5+.5*v
SCHEMES=['Gaussian','normalized_Gaussian','whole_sign','coordinate_sign','Haar_SO2',
         'identity','public_scalar_alternation','public_coordinate_alternation','public_R90_supplement']

def theta_matrix(A):
    T=np.zeros((8,8));R=.5*np.eye(2)
    for k in range(4):
        T[2*k:2*k+2,2*k:2*k+2]=np.eye(2)+R+A@R@A.T
        if k<3:
            T[2*k:2*k+2,2*k+2:2*k+4]=-R@A.T
            T[2*k+2:2*k+4,2*k:2*k+2]=-A@R
    return T

def deterministic_word(scheme,length=10):
    if scheme=='identity':return np.tile(np.eye(2),(length,1,1))
    if scheme=='public_scalar_alternation':return np.array([(-1.)**t*np.eye(2)for t in range(length)])
    if scheme=='public_coordinate_alternation':return np.array([np.diag([1.,(-1.)**t])for t in range(length)])
    if scheme=='public_R90_supplement':return np.tile(np.array([[0.,-1.],[1.,0.]]),(length,1,1))
    raise ValueError(scheme)

def first_Ms(scheme,n):
    if scheme=='whole_sign':
        signs=np.array(list(itertools.product([-1.,1.],repeat=5)))
        return signs[:,:,None,None]*np.eye(2)[None,None,:,:],dict(method='complete 32-word finite average',nodes=32)
    if scheme=='coordinate_sign':
        signs=np.array(list(itertools.product([-1.,1.],repeat=10))).reshape(-1,5,2)
        out=np.zeros((len(signs),5,2,2));out[:,:,0,0]=signs[:,:,0];out[:,:,1,1]=signs[:,:,1]
        return out,dict(method='complete 1024-word finite average',nodes=1024)
    if scheme in ['Gaussian','normalized_Gaussian']:
        u=qmc.Sobol(10,scramble=True,seed=20261114).random_base2(int(np.log2(n)))
        beta=norm.ppf(np.clip(u,1e-12,1-1e-12)).reshape(-1,5,2)
        out=np.eye(2)[None,None,:,:]+beta[:,:,:,None]*beta[:,:,None,:]
        if scheme=='normalized_Gaussian':out*=np.sqrt(NU)
        return out,dict(method='fixed-seed scrambled Sobol Gaussian integral diagnostic',nodes=n,seed=20261114,
                        endpoint_clip='[1e-12,1-1e-12]',integral_error_certified=False)
    if scheme=='Haar_SO2':
        u=qmc.Sobol(5,scramble=True,seed=20261115).random_base2(int(np.log2(n)))
        z=2*np.pi*u;out=np.empty((n,5,2,2))
        out[:,:,0,0]=np.cos(z);out[:,:,1,1]=np.cos(z)
        out[:,:,0,1]=-np.sin(z);out[:,:,1,0]=np.sin(z)
        return out,dict(method='fixed-seed scrambled Sobol angle integral diagnostic, iid SO(2)',nodes=n,
                        seed=20261115,not_O2_Haar=True,integral_error_certified=False)
    return deterministic_word(scheme,5)[None,:,:,:],dict(method='deterministic fixed word',nodes=1)

def path_Qs(A,scheme,n):
    Ms,meta=first_Ms(scheme,n);O=np.stack([np.linalg.matrix_power(A,j)for j in range(5)])
    maps=np.einsum('ntij,tjk->ntik',Ms,O)
    J=maps[:,1:]-np.einsum('ij,ntjk->ntik',A,maps[:,:-1])
    J=J.reshape(len(Ms),8,2)
    Qs=np.einsum('nri,rs,nsj->nij',J,np.linalg.inv(theta_matrix(A)),J)
    Qs=(Qs+Qs.transpose(0,2,1))/2
    return Qs,meta

def analytic_matrices(A,F,G,scheme):
    Os=[np.linalg.matrix_power(A,t)for t in range(L)]
    W=sum((O.T@O for O in Os),start=np.zeros((2,2)))
    Cs=[np.linalg.matrix_power(F,L-1-t)@G for t in range(L)]
    B0=sum((C@O for C,O in zip(Cs,Os)),start=np.zeros((2,2)))
    S0=B0.T@B0
    if scheme in ['identity','public_scalar_alternation','public_coordinate_alternation','public_R90_supplement']:
        M=deterministic_word(scheme,L)
        B=sum((C@Mt@O for C,Mt,O in zip(Cs,M,Os)),start=np.zeros((2,2)))
        S=B.T@B
    else:
        S=np.zeros((2,2))
        for C,O in zip(Cs,Os):
            X=C.T@C
            if scheme=='whole_sign':P=X
            elif scheme=='coordinate_sign':P=np.diag(np.diag(X))
            elif scheme=='Haar_SO2':P=np.trace(X)/2*np.eye(2)
            elif scheme in ['Gaussian','normalized_Gaussian']:P=X+np.trace(X)*np.eye(2)
            S+=O.T@P@O
        if scheme in ['Gaussian','normalized_Gaussian']:S+=4*S0
        if scheme=='normalized_Gaussian':S*=NU
    # Mean response in the complete first-five window. Cross terms vanish for
    # zero-mean independent coders; Gaussian's constant mean term is Z O=0.
    inv=np.linalg.inv(theta_matrix(A));Z=np.zeros((8,10))
    for t in range(4):Z[2*t:2*t+2,2*t:2*t+2]=-A;Z[2*t:2*t+2,2*t+2:2*t+4]=np.eye(2)
    P=Z.T@inv@Z;meanQ=np.zeros((2,2))
    if scheme in ['identity','public_scalar_alternation','public_coordinate_alternation','public_R90_supplement']:
        M=deterministic_word(scheme,5)
        J=Z@np.vstack([M[t]@Os[t]for t in range(5)])
        meanQ=J.T@inv@J
    else:
        for t in range(5):
            X=P[2*t:2*t+2,2*t:2*t+2]
            if scheme=='whole_sign':T=X
            elif scheme=='coordinate_sign':T=np.diag(np.diag(X))
            elif scheme=='Haar_SO2':T=np.trace(X)/2*np.eye(2)
            else:T=X+np.trace(X)*np.eye(2)
            meanQ+=Os[t].T@T@Os[t]
        if scheme=='normalized_Gaussian':meanQ*=NU
    return W,S,S0,meanQ

def solve_angle(Qs,W,S=None,grid=129):
    # W-orthonormal angular coordinates avoid arbitrary state scaling.
    ev,U=eigh(W);V=U@np.diag(1/np.sqrt(ev))@U.T
    if S is not None:
        upper=float(E*eigh(S,W,eigvals_only=True)[-1])
        if upper<D*(1-1e-12):return dict(feasible=False,actual_P3_damage_unreachable=True,Pd=None)
        Dm=V.T@S@V
    else:Dm=None
    def point(t,ret=False):
        u=V@np.array([np.cos(t),np.sin(t)])
        r2=E if S is None else D/float(u@S@u)
        if r2>E*(1+1e-9):return (np.inf,None)if ret else np.inf
        xi=np.sqrt(r2)*u
        lam=np.maximum(np.einsum('i,nij,j->n',xi,Qs,xi),0)
        val=float(np.mean(ncx2.sf(TAU,8,lam)))
        return (val,xi)if ret else val
    ts=list(np.linspace(0,np.pi,grid))
    if Dm is not None:
        center=np.trace(Dm)/2;a=(Dm[0,0]-Dm[1,1])/2;b=Dm[0,1];amp=np.hypot(a,b)
        if amp>0 and abs((D/E-center)/amp)<=1:
            ph=np.arctan2(b,a);ac=np.arccos(np.clip((D/E-center)/amp,-1,1))
            ts.extend([float(((ph+ac)/2)%np.pi),float(((ph-ac)/2)%np.pi)])
    ts=np.sort(np.unique(ts));vs=np.array([point(t)for t in ts]);candidates=[(v,float(t))for v,t in zip(vs,ts)if np.isfinite(v)]
    assert candidates,'P3 feasible but no sampled feasible direction'
    for i in np.argsort(vs)[:4]:
        if not np.isfinite(vs[i]):continue
        lo,hi=ts[max(0,i-1)],ts[min(len(ts)-1,i+1)]
        if hi>lo:
            opt=minimize_scalar(point,bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
            if np.isfinite(opt.fun):candidates.append((float(opt.fun),float(opt.x)))
    val,t=min(candidates);val,xi=point(t,True)
    return dict(feasible=True,Pd=val,initial_state=xi.tolist(),energy=float(xi@W@xi),
                expected_damage=(float(xi@S@xi)if S is not None else None),angle=t,
                angular_grid=grid,local_refinements=4,global_optimum_certified=False)

def run():
    out=dict(date='2026-11-14',baseline_commit='2d4db76',model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        estimator_unchanged=True,normal_fusion_unchanged=True,normal_MSE_same_as_original=True,
        L=L,E=E,D=D,s=4,first_full_window='attacked packets t=0..4; score at t=4',
        alpha_test=alpha,total_event_FA_union_bound=.05,finite_test_count=15,threshold=TAU,
        statistic_df=8,physical_noise_samples=0,
        actual_Pd='mean of conditional ncx2 survival functions, not survival at E Lambda',
        QMC_is_not_integral_error_certificate=True,normalization_nu=NU,models={})
    for name,m in data['models'].items():
        A=np.array(m['A']);nd=m['nodes'][1];F=np.array(nd['posterior_F']);G=np.array(nd['G'])
        rows={}
        for scheme in SCHEMES:
            W,S,S0,meanQ=analytic_matrices(A,F,G,scheme)
            eig,U=eigh(S,W);xi=np.sqrt(E)*U[:,-1]
            geig,V=eigh(meanQ,W)
            levels=[1024,4096,8192]if scheme in ['Gaussian','normalized_Gaussian','Haar_SO2']else[0]
            probes=[]
            for n in levels:
                Qs,meta=path_Qs(A,scheme,n)
                pp=solve_angle(Qs,W)
                pot=solve_angle(Qs,W,S0)
                phys=solve_angle(Qs,W,S)
                probes.append(dict(nodes=meta['nodes'],integration=meta,E10_min_power=pp,
                    P3_pot=pot,P3_phys=phys,mean_ncp_matrix_error=np.max(abs(Qs.mean(axis=0)-meanQ)).item()))
            rows[scheme]=dict(W=W.tolist(),S=S.tolist(),S0=S0.tolist(),expected_ncp_matrix=meanQ.tolist(),
                min_expected_ncp_E10=float(E*geig[0]),min_expected_ncp_direction=(np.sqrt(E)*V[:,0]).tolist(),
                P1_expected_terminal_damage=float(E*eig[-1]),P1_initial_state=xi.tolist(),
                P3_phys_feasible=bool(E*eig[-1]>=D),probes=probes,
                finite_event_risk_computed=False)
            out['models'][name]=dict(A=A.tolist(),F=F.tolist(),G=G.tolist(),schemes=rows)
            (H/'baseline_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(name,scheme,'E Lambda min',E*geig[0],'P1',E*eig[-1],
                  'Pd',probes[-1]['E10_min_power']['Pd'],flush=True)
    (H/'baseline_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':run()
