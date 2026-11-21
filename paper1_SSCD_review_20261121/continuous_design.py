"""One-dimensional signed-zeta SSCD screen with p=2 continuum attack set.

For a two-dimensional effective attack parameter, the exact energy shell is a
circle after whitening by its coded-energy Gram. This module bounds the inner
continuous-angle minimum and the outer zeta search. Floating-point norms and
quadrature are not outward-rounded; results are numerical certificates with
an explicitly reported residual gap, not formal interval-arithmetic proofs.
"""
from __future__ import annotations
import heapq
import numpy as np
from scipy.linalg import cholesky
from sscd import watson_stats, theta_from_zeta, dependence_interval


def _sym_norm(A, kind):
    return float(np.linalg.norm((A+A.T)/2, ord=kind))


def _covariance_polynomial(d,q):
    """Return V_h with Cov(X)=sum zeta**h V_h, for fixed block vectors d."""
    d=np.asarray(d,float);L,m=d.shape
    V=[np.zeros((L*m,L*m)) for _ in range(L)]
    for ell in range(L):
        sl=slice(ell*m,(ell+1)*m);x=d[ell]
        V[0][sl,sl]=q*q*(x@x*np.eye(m)+np.outer(x,x))
        for r in range(ell+1,L):
            h=r-ell;sr=slice(r*m,(r+1)*m);y=d[r]
            C=q*q*m/(m+2.0)*((x@y)*np.eye(m)+np.outer(y,x)-(2/m)*np.outer(x,y))
            V[h][sl,sr]=C;V[h][sr,sl]=C.T
    return V


def _residual_covariance_polynomials(T_blocks,Z,Theta,energy_gram,energy,q):
    T=[np.asarray(x,float) for x in T_blocks];Z=np.asarray(Z,float);Theta=np.asarray(Theta,float)
    E=np.asarray(energy_gram,float);L=len(T);m,p=T[0].shape
    if p!=2 or E.shape!=(2,2) or Z.shape[1]!=L*m or Theta.shape!=(Z.shape[0],Z.shape[0]):
        raise ValueError('expected p=2, compatible parity and nominal covariance dimensions')
    if np.linalg.norm(Z@np.vstack(T))>1e-7:
        raise ValueError('detector attack subspace must be parity-stealth for this covariance-only objective')
    ev,U=np.linalg.eigh((E+E.T)/2)
    if ev.min()<=0:raise ValueError('coded-energy Gram must be positive definite on effective parameter space')
    energy_basis=np.sqrt(energy)*(U/np.sqrt(ev))@U.T
    chol=cholesky(Theta,lower=True)
    def whiten(C):
        left=np.linalg.solve(chol,C)
        return np.linalg.solve(chol,left.T).T
    coeff=[]
    basis_cov=[]
    for vector in (energy_basis[:,0],energy_basis[:,1],energy_basis[:,0]+energy_basis[:,1]):
        d=np.stack([t@vector for t in T])
        basis_cov.append([whiten(Z@V@Z.T) for V in _covariance_polynomial(d,q)])
    for h in range(L):
        M11,M22,Msum=(basis_cov[j][h] for j in range(3))
        cross=(Msum-M11-M22)/2
        coeff.append(((M11+M22)/2,(M11-M22)/2,cross))
    return coeff


def parity_energy_coefficients(T_blocks,Z,Theta,energy_gram,energy,q=.5):
    """Public p=2 energy-shell covariance precomputation interface.

    energy_gram must describe the entire declared attack, even when T_blocks
    contains only one internal detector window. Columns must be the same
    effective attack-parameter coordinates as the state-risk polynomials.
    """
    return _residual_covariance_polynomials(T_blocks,Z,Theta,energy_gram,energy,q)


def state_risk_polynomials(T_blocks,K_blocks,W,q=.5,lam=.5):
    """Return exact polynomial Q_h for fixed common-G linear output maps."""
    T=[np.asarray(x,float) for x in T_blocks]
    K=[np.asarray(x,float) for x in K_blocks]
    p=T[0].shape[1];L=len(T);Kmap=np.concatenate(K,axis=1);W=np.asarray(W,float)
    eye=np.eye(p);values={}
    for i in range(p):
        for j in range(i,p):
            x=eye[i] if i==j else eye[i]+eye[j]
            d=np.stack([t@x for t in T]);mean=(lam+q)*Kmap@d.reshape(-1)
            risk=np.array([np.trace(W@Kmap@v@Kmap.T) for v in _covariance_polynomial(d,q)])
            risk[0]+=mean@W@mean
            values[i,j]=risk
    coeff=[np.zeros((p,p)) for _ in range(L)]
    for h in range(L):
        for i in range(p):
            coeff[h][i,i]=values[i,i][h]
            for j in range(i):
                coeff[h][i,j]=coeff[h][j,i]=.5*(values[j,i][h]-values[i,i][h]-values[j,j][h])
    return coeff


def _dg(B):
    B=(B+B.T)/2
    eig=np.linalg.eigvalsh(B)
    if eig.min() < -1e-8*max(1.,np.linalg.norm(B,2)):
        raise ArithmeticError('whitened covariance polynomial is not PSD')
    eig=np.maximum(eig,0.)
    return float(.5*np.sum(eig-np.log1p(eig)))


def angle_minimum(coeff,zeta,points=256):
    """Return analytic angle grid upper and smoothness-based lower bound."""
    phi=np.arange(points)*np.pi/points
    cos=np.cos(2*phi);sin=np.sin(2*phi)
    B=np.zeros_like(coeff[0][0])
    Bc=np.zeros_like(B);Bs=np.zeros_like(B)
    for h,(a,c,s) in enumerate(coeff):
        B+=zeta**h*a;Bc+=zeta**h*c;Bs+=zeta**h*s
    vals=np.array([_dg(B+Bc*x+Bs*y) for x,y in zip(cos,sin)])
    upper=float(vals.min())
    Rn=np.sqrt(_sym_norm(Bc,'nuc')**2+_sym_norm(Bs,'nuc')**2)
    Rf2=_sym_norm(Bc,'fro')**2+_sym_norm(Bs,'fro')**2
    Mphi=2*Rn+2*Rf2
    delta=np.pi/(2*points)
    lower=max(0.,upper-.5*Mphi*delta**2)
    return dict(lower=lower,upper=upper,grid_phi=float(phi[vals.argmin()]),
                curvature_bound=float(Mphi),cover_radius=float(delta))


def _poly_value(coeff,z):
    out=np.zeros_like(coeff[0]);
    for a in reversed(coeff):out=out*z+a
    return (out+out.T)/2


def screen_continuum_2d(detector_coeff,state_coeff,energy_gram,energy,
                        dependence_budget,state_risk_budget,window_length,
                        theta_cap=20.,angle_points=256,zeta_tolerance=5e-4,
                        evaluation_cap=1500,measurement_dimension=2,
                        additional_detector_coeff=(),directionwise_tolerance=None):
    """Signed SSCD with exact p=2 energy shell and bounded inner-angle gap.

    detector_coeff: tuple of (A_h,Bc_h,Bs_h), whitened covariance matrix
      coefficients after the energy-sphere map.
    state_coeff: Q_h coefficients in attack parameter coordinates.
    The search is global-gap screened over a compact theta domain, but uses
    floating-point bounds, not formal interval arithmetic. An unresolved gap
    or isolated feasible point is reported rather than silently certified.
    """
    from sscd import THETA_CAP
    if abs(theta_cap-THETA_CAP)>1e-12:
        raise ValueError('theta_cap must match the validated kernel domain')
    m=measurement_dimension
    domain=dependence_interval(m,window_length,dependence_budget)
    lo,hi=domain['zeta_interval'];Qcoeff=[np.asarray(q,float) for q in state_coeff]
    E=np.asarray(energy_gram,float);ev,U=np.linalg.eigh((E+E.T)/2)
    if ev.min()<=0:raise ValueError('energy Gram must be positive definite')
    Einv=(U/np.sqrt(ev))@U.T
    if directionwise_tolerance is not None and directionwise_tolerance<0:
        raise ValueError('directionwise_tolerance must be nonnegative')
    Qiid=Qcoeff[0]
    def state_risk(z):
        Q=_poly_value(Qcoeff,z)
        return float(energy*np.linalg.eigvalsh(Einv@Q@Einv)[-1])
    def state_directionwise(z):
        return float(energy*np.linalg.eigvalsh(Einv@(_poly_value(Qcoeff,z)-Qiid)@Einv)[-1])
    r=max(abs(lo),abs(hi));Lr=0.
    for h in range(1,len(Qcoeff)):
        Lr+=energy*h*r**(h-1)*np.linalg.norm(Einv@Qcoeff[h]@Einv,2)
    detector_families=[detector_coeff]+list(additional_detector_coeff)
    Ld=0.
    for family in detector_families:
        single=0.
        for h,(a,c,s) in enumerate(family):
            if h==0:continue
            single+=.5*h*r**(h-1)*(_sym_norm(a,'nuc')+
                                  np.sqrt(_sym_norm(c,'nuc')**2+_sym_norm(s,'nuc')**2))
        Ld=max(Ld,single)
    cache={}
    def evaluate(z):
        key=float(z)
        if key not in cache:
            stat=watson_stats(m,theta_from_zeta(m,key))
            dependence=(window_length-1)*stat['mi_nats']
            risk=state_risk(key)
            directionwise=state_directionwise(key)
            parts=[angle_minimum(family,key,angle_points) for family in detector_families]
            inner=dict(lower=min(p['lower'] for p in parts),upper=min(p['upper'] for p in parts),parts=parts)
            cache[key]=dict(zeta=key,theta=theta_from_zeta(m,key),
                            dependence=dependence,state_risk=risk,inner=inner,
                            state_directionwise_change=directionwise,
                            feasible=dependence<=dependence_budget+1e-11 and risk<=state_risk_budget+1e-10
                            and (directionwise_tolerance is None or directionwise<=directionwise_tolerance+1e-10))
        return cache[key]
    iid=evaluate(0.)
    if not iid['feasible']:raise ValueError('IID baseline is infeasible; revise state/dependence budget explicitly')
    best=iid
    heap=[];serial=0;terminal_upper=-np.inf
    def push(a,b):
        nonlocal serial,best
        c=(a+b)/2;half=(b-a)/2;risk=state_risk(c);Lrisk=Lr*half
        if risk-Lrisk>state_risk_budget+1e-10:return
        if directionwise_tolerance is not None and state_directionwise(c)-Lrisk>directionwise_tolerance+1e-10:return
        mid=evaluate(c)
        if mid['feasible'] and mid['inner']['lower']>best['inner']['lower']:best=mid
        ub=mid['inner']['upper']+Ld*half
        heapq.heappush(heap,(-ub,serial,a,b));serial+=1
    for z in (lo,hi):
        endpoint=evaluate(z)
        if endpoint['feasible'] and endpoint['inner']['lower']>best['inner']['lower']:best=endpoint
    push(lo,hi)
    while heap and len(cache)<evaluation_cap:
        negub,_,a,b=heapq.heappop(heap);ub=-negub
        if ub<=best['inner']['lower']+zeta_tolerance:
            terminal_upper=max(terminal_upper,ub);continue
        c=(a+b)/2;half=(b-a)/2;mid=evaluate(c)
        if mid['feasible'] and mid['inner']['lower']>best['inner']['lower']:best=mid
        if half<=zeta_tolerance/(Ld+1e-15):
            terminal_upper=max(terminal_upper,ub);continue
        push(a,c);push(c,b)
    remaining=max([terminal_upper,best['inner']['upper']]+[-x[0] for x in heap])
    gap=max(0.,remaining-best['inner']['lower'])
    optimized=best
    # Require a bounded strict proxy improvement, not just a better sampled
    # minimum that may be explained by the inner angular discretization gap.
    if best['inner']['lower']<=iid['inner']['upper']+1e-10:
        best=iid;policy_reason='retain IID: no strict lower-bound gain above IID upper bound'
    else:
        policy_reason='select coupling: candidate lower bound exceeds IID upper bound'
    return dict(status='screened p=2 continuum candidate',best=best,iid=iid,
                optimization_candidate=optimized,policy_reason=policy_reason,
                remaining_objective_gap=float(gap),evaluation_count=len(cache),
                selected_policy_regret_upper=float(max(0.,remaining-best['inner']['lower'])),
                evaluation_cap=evaluation_cap,converged=bool(gap<=zeta_tolerance),
                bounds=dict(theta=domain['theta_interval'],zeta=domain['zeta_interval'],
                            dependence_budget=dependence_budget,state_risk_budget=state_risk_budget),
                directionwise_tolerance=directionwise_tolerance,
                lipschitz=dict(DG_zeta=float(Ld),state_risk_zeta=float(Lr)),
                caveat='Floating-point Lipschitz screen only; not outward-rounded interval certification or arbitrary-p attack guarantee.')
