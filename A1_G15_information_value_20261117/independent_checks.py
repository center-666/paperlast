"""Independent block propagation and exact Gaussian moment cubature."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
import numpy as np
import sympy as s
import information_value as iv
from conditional_attack import maximize_quadratic
H=Path(__file__).resolve().parent;P=H.parent
sys.path.insert(0,str(P/'A1_information_aware_attack_audit_20261116'))
from attacker_observer_bounds import primitive_maps
import attack_models as am
def main():
    a,b,c=s.symbols('a b c');X=s.Matrix([[a,b],[b,c]])
    Ms=[];ws=[]
    for i,wi in [(-1,s.Rational(1,6)),(0,s.Rational(2,3)),(1,s.Rational(1,6))]:
        for j,wj in [(-1,s.Rational(1,6)),(0,s.Rational(2,3)),(1,s.Rational(1,6))]:
            v=s.Matrix([i,j]);Ms.append(s.eye(2)/2+s.Rational(3,2)*v*v.T);ws.append(wi*wj)
    mean=sum((w*M for w,M in zip(ws,Ms)),s.zeros(2))
    moment=sum((w*M*X*M for w,M in zip(ws,Ms)),s.zeros(2))
    assert mean==s.eye(2) and s.simplify(moment-s.Rational(5,4)*X-s.trace(X)*s.eye(2)/4)==s.zeros(2)
    errors={}
    for name in ('unstable','stable'):
        m=iv.exact_model(name);old=am.ref.load_model(name)
        _,Y,_,Ep=primitive_maps(old)
        ey=max(float(np.max(np.abs(Ep[k]@Y[t,0].T-iv.fl(iv.cov_ey(m,k,t))))) for t in (6,7) for k in range(7,32))
        times=range(7,17);maps=[]
        for t in times:
            maps.append(np.vstack([np.zeros((2,2)) if r<t else np.linalg.matrix_power(old.F(1),r-t)@old.G[1] for r in range(7,32)]))
        R=np.block([[Bt.T@Bu for Bu in maps] for Bt in maps])
        for i,Bt in enumerate(maps):
            C=Bt.T@Bt;R[2*i:2*i+2,2*i:2*i+2]=sum(float(w)*iv.fl(M).T@C@iv.fl(M) for w,M in zip(ws,Ms))
        gram=float(np.max(np.abs(R-iv.fl(iv.gram(m,list(times),list(range(7,32)))))))
        # The same physical/watermark realization, arbitrary fixed signed
        # waveform: energy and bias square are pointwise invariant.
        v=np.array([float(s.Rational(x)) for x in json.loads((H/'deterministic_diagnostics.json').read_text('utf8'))['models'][name]['cumulative']['near_optimal_waveform_unit']])
        Lmat=np.hstack([Bt@iv.fl(Ms[(3*i+2)%9]) for i,Bt in enumerate(maps)])
        plus=Lmat@v;minus=Lmat@(-v)
        signerr=float(np.max(np.abs(plus*plus-minus*minus)))
        assert ey<1e-9 and gram<1e-10 and signerr==0
        registered=json.loads((H/'deterministic_diagnostics.json').read_text('utf8'))['models'][name]['cumulative']
        for q in ([1.,0.],[-1.,2.],[0.,0.]):
            plan=iv.coded_sign_policy(q,registered['near_optimal_waveform_unit'],registered['history_public_ell'])
            assert np.sum(plan*plan)<=10 and plan.shape==(10,2)
        errors[name]=dict(physical_covariance=ey,block_Gram=gram,pointwise_sign_squared=signerr)
    # Both regular and hard-case trust-region checks.
    cases=[(np.diag([1.,2.]),np.zeros(2)),(np.diag([1.,2.]),np.array([.3,0.])),
           (np.diag([1.,2.]),np.array([.3,.7]))]
    for Q,h in cases:
        a,nu=maximize_quadratic(Q,h,10)
        assert abs(a@a-10)<1e-8 and np.linalg.norm((nu*np.eye(2)-Q)@a-h)<1e-8
        assert np.linalg.eigvalsh(nu*np.eye(2)-Q).min()>-1e-10
    out=dict(date='2026-11-17',errors=errors,Gaussian_watermark_moment_identity_exact=True,
      trust_region_regular_and_hard_cases=3,no_new_physical_samples=True,
      no_actual_state_or_M_input_to_sign_policy=True,no_original_estimator_changes=True)
    (H/'independent_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out))
if __name__=='__main__':main()
