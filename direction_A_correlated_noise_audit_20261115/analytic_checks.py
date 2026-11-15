"""Independent G0 analytic checks only. No estimator redesign, sampling, or new test."""
import hashlib, json, pathlib
import numpy as np
import sympy as S
from scipy.stats import chi2
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=HERE.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
def rat(x): return S.Rational(str(x))
def mat(x): return S.Matrix([[rat(v) for v in row] for row in x])
def floats(x): return np.array(x.tolist(),float)
def exact_lyap(F,O):
    x,y,z=S.symbols('x y z'); P=S.Matrix([[x,y],[y,z]])
    d=P-F*P*F.T-O
    sol=S.solve([d[0,0],d[0,1],d[1,1]],(x,y,z))
    ans=P.subs(sol)
    assert ans-F*ans*F.T==O
    return ans
def main():
    model=json.loads(SOURCE.read_text(encoding='utf-8'))
    result={'scope':'G0 finite analytic diagnostics; fixed decimal gains interpreted as exact rationals; no DARE recomputation; no physical samples',
       'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'physical_samples':0}
    H=[mat(h) for h in model['H']]; Q=mat(model['Q']); I=S.eye(2)
    rows=[]; witnesses={}
    for name,mod in model['models'].items():
        A=mat(mod['A']); target=mod['nodes'][1]; K=mat(target['K']); G=mat(target['G'])
        neigh=target['neighbors']; J=I-K*H[1]-G*sum((H[j] for j in neigh),S.zeros(2))
        F=J*A
        # Rebuild F from original common-G equation; saved rounded product is not treated as identical.
        Gamma=S.zeros(2,10)
        for j in range(5):
            block=K if j==1 else G if j in neigh else S.zeros(2)
            Gamma[:,2*j:2*j+2]=block
        P0=None; delta=None; diagnostics=[]
        for rho in [S.Rational(-1,4),S.Rational(-1,5),S.Rational(0),S.Rational(1,2),S.Rational(1)]:
            C=(1-rho)*S.eye(5)+rho*S.ones(5)
            # eigenvalues 1-rho (4-fold), 1+4rho (1-fold), hence global PSD.
            assert 1-rho>=0 and 1+4*rho>=0
            Rjoint=S.kronecker_product(C,S.eye(2))/2
            Omega=J*Q*J.T+Gamma*Rjoint*Gamma.T
            P=exact_lyap(F,Omega)
            diagnostics.append({'rho':str(rho),'joint_eigenvalues':[float((1-rho)/2),float((1+4*rho)/2)],
                'posterior_trace':float(S.trace(P)), 'prediction_trace':float(S.trace(A*P*A.T+Q)),
                'aggregate_R':[[float(v) for v in row] for row in (S.Matrix.vstack(S.Matrix.hstack(I/2,rho*I),S.Matrix.hstack(rho*I,(1+rho)*I))).tolist()],
                'lyapunov_residual_exact_zero':True})
            if rho==0: P0=P
            if rho==1: P1=P
        for r in diagnostics: r['relative_to_independent_percent']=100*(r['posterior_trace']/float(S.trace(P0))-1)
        # Fixed F and attack input are correlation-independent; P is affine in rho.
        rho=S.symbols('rho',real=True)
        Rrho=S.kronecker_product((1-rho)*S.eye(5)+rho*S.ones(5),I)/2
        O=J*Q*J.T+Gamma*Rrho*Gamma.T
        assert S.simplify((P0+rho*(P1-P0))-F*(P0+rho*(P1-P0))*F.T-O)==S.zeros(2)
        # Attribution intersection: two neighbor single-link autonomous attack maps for L=1 and 2.
        ranks=[]
        for L in [1,2,10]:
            T0=S.Matrix.vstack(*(H[0]*A**t for t in range(L)))
            T2=S.Matrix.vstack(*(H[2]*A**t for t in range(L)))
            rank=T0.row_join(-T2).rank()
            ranks.append({'L':L,'combined_rank':rank,'intersection_dimension':T0.rank()+T2.rank()-rank})
        rows.append({'name':name,'F_spectral_radius':max(abs(np.linalg.eigvals(floats(F)))),
             'F_norm2':np.linalg.norm(floats(F),2),'saved_F_difference':float(np.linalg.norm(floats(F)-np.array(target['posterior_F']),2)),
             'rho_cases':diagnostics,'posterior_affine_in_rho_exact':True,'actual_neighbor_template_intersections':ranks})
        witnesses[name]={'F_rebuilt':[[str(v) for v in row] for row in F.tolist()],
             'K':target['K'],'G':target['G'], 'P_rho0_exact':[[str(v) for v in row] for row in P0.tolist()],
             'P_rho1_exact':[[str(v) for v in row] for row in P1.tolist()]}
    result['aggregate_model']=rows
    # Scalar x_{k+1}=x_k+w_k, var(w)=1, var(v_i)=var(v_j)=1/2.
    # r_t=y_j,k-y_j,k-1; e_sp=y_j,k-y_i,k. State cancels in e_sp.
    tau=chi2.isf(.05,1); scalar=[]
    for r in [S.Rational(-1,2),S.Rational(0),S.Rational(1,2)]:
        temporal=S.Integer(2); spatial=1-r; cross=(1-r)/2
        conditional=spatial-cross**2/temporal
        scalar.append({'rho':str(r),'time_variance':float(temporal),'space_variance':float(spatial),
          'time_space_covariance':float(cross),'conditional_space_variance':float(conditional),
          'nominal_independence_spatial_FAR':float(chi2.sf(tau/float(spatial),1)),
          'covariance_envelope_2_FAR':float(chi2.sf(2*tau/float(spatial),1))})
        assert temporal*spatial-cross**2>0
    result['scalar_checks']=scalar
    result['directional_counterexample']={'Rii':'I2','Rjj':'I2','Rij':'diag(1/2,-1/2)',
        'global_joint_eigenvalues':[.5,.5,1.5,1.5], 'actual_spatial_covariance':[[1,0],[0,3]],'independent_covariance':[[2,0],[0,2]],
        'unit_attack_noncentralities':{'x':1.,'y':1/3},'independent_noncentrality':.5}
    # Gaussian aggregate cancellations do not require sensor independence.
    C=S.Matrix([[1,S.Rational(1,5),-S.Rational(1,5)],[S.Rational(1,5),1,0],[-S.Rational(1,5),0,1]])
    assert all(C[:k,:k].det()>0 for k in range(1,4))
    Sagg=S.Matrix([[1,0,0],[0,1,1]])
    assert Sagg*C*Sagg.T==S.diag(1,2)
    result['aggregation_cancellation_counterexample']={'C':[[str(v) for v in row] for row in C.tolist()],
       'aggregate_covariance':[[1,0],[0,2]],'leading_principal_minors':[str(C[:k,:k].det()) for k in range(1,4)]}
    # Matched conditional identity for a fixed mean [0,d] and covariance Sigma.
    t,e,d=S.symbols('t e d'); V=S.Matrix([[2,S.Rational(1,2)],[S.Rational(1,2),1]])
    x=S.Matrix([t,e]); mu=S.Matrix([0,d]); inv=V.inv()
    ll=S.expand(mu.dot(inv*x)-mu.dot(inv*mu)/2)
    c=S.Rational(1,4); Vc=S.Rational(7,8)
    assert S.simplify(ll-(d*(e-c*t)/Vc-d*d/(2*Vc)))==0
    result['joint_matched_conditional_identity_exact']=True
    (HERE/'analytic_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'exact_rational_witnesses.json').write_text(json.dumps(witnesses,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'models':rows,'scalar':scalar},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
