"""Candidate II necessity audit: identities and fixed-model arithmetic, no new algorithm.

Uses only the already frozen original-compatible common-G five-node model.
No old script is imported, no physical simulation and no random sampling.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import sympy as sp
from scipy.linalg import eigvalsh

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'original_estimator_gate_20261114/five_node_aggregate_model.json'

def smat(x): return sp.Matrix([[sp.Rational(str(z)) for z in row] for row in x])
def arr(x): return np.array(x.tolist(), dtype=float)
def pd2(M): return M == M.T and M[0,0] > 0 and M.det() > 0

def exact_linear_identities():
    # Single scalar common-G filter: a false reconstruction can increase MSE.
    R = sp.Rational(1, 2); K = sp.Rational(1,5); G = sp.Rational(2,5)
    baseline = (K*K+G*G)*R
    corrected = (K+G)**2*R
    # Current residual y_j-y_i estimates the scalar attack with noise v_j-v_i.
    xi_variance = 2*R
    eps_error_cross = (G-K)*R
    increment = G**2*xi_variance - 2*G*eps_error_cross
    assert corrected-baseline == increment == sp.Rational(2,25)
    assert baseline == sp.Rational(1,10) and corrected == sp.Rational(9,50)
    # A scalar replay identity, valid for arbitrary numerical data and fitted xi.
    f,g,dh0,dh1,dh2 = sp.symbols('f g dh0 dh1 dh2')
    v0,v1,v2,z0 = sp.symbols('v0 v1 v2 z0')
    old=z0
    new=z0
    for v,d in [(v0,dh0),(v1,dh1),(v2,dh2)]:
        old=f*old+g*v
        new=f*new+g*(v-d)
    convolution=g*(f*f*dh0+f*dh1+dh2)
    assert sp.expand(old-new-convolution) == 0
    return {'false_alarm_counterexample': {
      'marginal_noise_variance':str(R),'K':str(K),'G':str(G),
      'healthy_MSE':str(baseline),'corrected_healthy_MSE':str(corrected),
      'increment':str(increment),'attack_damage':str(G**2)+' * xi^2',
      'compensation_helps_only_if_xi_squared_gt':str(increment/G**2),
      'scope':'one-packet scalar common-G example; no assertion about original five-node optimum'},
      'fixed_linear_history_replay_identity_exact':True}

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'reconstruction_checks.json')
    args=parser.parse_args()
    model=json.loads(SOURCE.read_text(encoding='utf-8'))
    H=[smat(x) for x in model['H']]
    outputs={}
    for name,m in model['models'].items():
        A=smat(m['A']); node=m['nodes'][1]; K=smat(node['K']); G=smat(node['G'])
        J=sp.eye(2)-K*H[1]-G*(H[0]+H[2]); F=J*A
        Fdrop=(J+G*H[0])*A
        c = sp.Rational(3,25) if name=='unstable' else sp.Rational(1,10)
        cd=sp.Rational(21,50) if name=='unstable' else sp.Rational(33,100)
        assert pd2(c*c*sp.eye(2)-F.T*F)
        assert pd2(cd*cd*sp.eye(2)-Fdrop.T*Fdrop)

        # Unknown-onset ambiguity only for observations strictly inside both attacks.
        # Both attacks have length 10; first starts 0, second 2. Same overlap 2..9.
        xi=sp.Matrix([sp.Rational(1,100),sp.Rational(2,100)])
        eta=A**2*xi
        for k in range(2,10):
            assert H[0]*A**k*xi == H[0]*A**(k-2)*eta
        assert H[0]*xi != sp.zeros(2,1)
        e1=sum((H[0]*A**t*xi).dot(H[0]*A**t*xi) for t in range(10))
        e2=sum((H[0]*A**t*eta).dot(H[0]*A**t*eta) for t in range(10))
        assert 0<e1<10 and 0<e2<10

        # No-encoding template support/chain identifiability on full known intervals.
        rank_rows=[]
        for L in [1,2,10]:
            T0=sp.Matrix.vstack(*(H[0]*A**t for t in range(L)))
            T2=sp.Matrix.vstack(*(H[2]*A**t for t in range(L)))
            rank_rows.append({'L':L,'rank_T0':T0.rank(),'rank_T2':T2.rank(),
              'rank_combined':T0.row_join(-T2).rank(),
              'intersection_dimension':T0.rank()+T2.rank()-T0.row_join(-T2).rank()})

        # Standard known-support GLS diagnostic; no attack detector or correction run.
        An,Fn,Gn,Kn=map(arr,[A,F,G,K]); D=arr(H[0]*H[1].inv())
        R0=np.eye(2)*.5; R1=np.eye(2)*.5; Theta=R0+D@R1@D.T
        gls=[]
        for L in [1,10]:
            blocks=[arr(H[0]*A**t) for t in range(L)]
            T=np.vstack(blocks); V=np.kron(np.eye(L),Theta)
            V_inv=np.linalg.inv(V); information=T.T@V_inv@T
            Xi=np.linalg.inv(information); Lgls=Xi@T.T@V_inv
            assert np.linalg.norm(Lgls@T-np.eye(2))<1e-12
            B=np.zeros((2,2))
            for Tt in blocks: B=Fn@B+Gn@Tt
            W=T.T@T
            emax=10*eigvalsh(B.T@B,W)[-1]
            cross_eta_error=np.vstack([(R0@Gn.T-D@R1@Kn.T)@np.linalg.matrix_power(Fn,L-1-t).T for t in range(L)])
            cross_xi_error=Lgls@cross_eta_error
            penalty=np.trace(B@Xi@B.T)-2*np.trace(B@cross_xi_error)
            gls.append({'L':L,'source_template_rank':int(np.linalg.matrix_rank(T)),
              'GLS_parameter_covariance':Xi.tolist(),
              'known_support_no_alarm_selection':True,
              'normal_error_GLS_cross_covariance':cross_xi_error.tolist(),
              'unconditional_compensation_MSE_increment_over_healthy':float(penalty),
              'not_a_detector_conditioned_result':True,
              'max_stop_bias_squared_at_E10_no_watermark':float(emax),
              'relative_tail_squared_bounds_after_1_2_3_updates':[float(c**(2*r)) for r in [1,2,3]]})
        outputs[name]={
          'model_gain_regime':'frozen nominal K/G; F independently rebuilt from original common-G formula',
          'F_rebuilt':arr(F).tolist(),'F_saved_difference_norm':float(np.linalg.norm(arr(F)-np.array(node['posterior_F']),2)),
          'F_spectral_radius':float(max(abs(np.linalg.eigvals(arr(F))))),'F_norm2_float':float(np.linalg.norm(arr(F),2)),
          'strict_norm_upper_bound_exact_rational':str(c),
          'norm_upper_bound_Sylvester_certificate_exact':True,
          'target_rejection_F_norm2_float':float(np.linalg.norm(arr(Fdrop),2)),
          'target_rejection_norm_upper_bound_exact_rational':str(cd),
          'target_rejection_norm_certificate_exact':True,
          'onset_interior_ambiguity_example':{'first_start':0,'second_start':2,'length_both':10,
            'xi':[str(x) for x in xi],'eta':[str(x) for x in eta],
            'equal_observed_indices':[2,9],'first_total_energy':float(e1),'second_total_energy':float(e2),
            'full_prehistory_distinguishes':True},
          'no_watermark_chain_template_ranks':rank_rows,'GLS_analytic_diagnostics':gls}
    result={'scope':'Candidate II necessity screen only; no new estimator/recovery algorithm or MC',
      'input_canonical_json_sha256':hashlib.sha256(json.dumps(model,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest(),
      'input':str(SOURCE),'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'physical_samples':0,'identities':exact_linear_identities(),'models':outputs,
      'limits':['Frozen K/G decimals treated as exact rationals for norm certificates; no physical uncertainty certification',
        'GLS numerics assume correct known onset/link/length and fixed window, no selection on alarm',
        'Tail certificates require common fixed admission path; data-dependent isolation changes dynamics',
        'No method novelty or whole-field impossibility asserted']}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'input_sha256':result['input_sha256'],
      'certificates':{k:{'F_norm_bound':v['strict_norm_upper_bound_exact_rational'],
        'Fdrop_norm_bound':v['target_rejection_norm_upper_bound_exact_rational'],
        'GLS':v['GLS_analytic_diagnostics']} for k,v in outputs.items()},
      'physical_samples':0},ensure_ascii=False))

if __name__=='__main__':main()
