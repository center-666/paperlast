"""Exact rational spectral and positive information-gain certificates.

Frozen decimal K/G are rational inputs; this is not DARE/physical uncertainty certification.
"""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
import sympy as s
import numpy as np
import information_value as iv
H=Path(__file__).resolve().parent
def enc(M):return [[str(v) for v in row] for row in M.tolist()]
def cert_spectral(R):
    # Reconstruct the whole eigendecomposition as rationals, then bound BOTH
    # its Gram error and its absolute reconstruction error by exact sums.
    eig,V=np.linalg.eigh(iv.fl(R));Vr=iv.sm(V.tolist());D=s.diag(*[s.Rational(str(v)) for v in eig])
    err=R-Vr*D*Vr.T;gram=Vr.T*Vr-s.eye(len(eig))
    err2=sum(v*v for v in err);gram2=sum(v*v for v in gram)
    eps=s.Rational(1,10**12)
    assert err2<eps**2 and gram2<eps**2
    assert min(D.diagonal())>0
    upper=max(D.diagonal())*(1+eps)+eps
    return upper,dict(eigenvalues=[str(v) for v in D.diagonal()],
      rational_eigenvectors=enc(Vr),reconstruction_frobenius_squared=str(err2),
      gram_frobenius_squared=str(gram2),certified_epsilon=str(eps),
      positive_D=True,lambda_upper_exact=str(upper),
      proof='R=Vr D VrT+error; lambda_max(Vr D VrT)<=max(D)*(1+||VrT Vr-I||F)')
def main():
    data={};npdata={}
    for name in ('unstable','stable'):
        print('Certifying '+name,flush=True);m=iv.exact_model(name);rows={}
        for label,times,ends in [('single',[7],[7]),('cumulative',list(range(7,17)),list(range(7,32)))]:
            R=iv.gram(m,times,ends);upper,spec=cert_spectral(R)
            _,V=np.linalg.eigh(iv.fl(R));v=iv.unit_rational(V[:,-1])
            q=(v.T*R*v)[0];c=sum(iv.coefficients(m,times,ends,v,6),s.zeros(1,2)).T
            py=m['Px'][6]+m['R'][0]
            B=s.Integer(math.ceil(math.sqrt(float(s.trace(py)))))
            assert B*B>=s.trace(py)
            # sqrt(2/pi)>.79; sqrt(E)>3; E[.5/(.5+r)]>=1/3.
            # |one c coordinate|<=||c||, sqrt(lambda_max Py)<=B.
            ci=max(c,key=abs);cross_lower=s.Rational(79,100)*abs(ci)/(3*B)
            gain_lower=6*cross_lower  # 2sqrt(10)>6
            H_lower=10*q+gain_lower;OL_upper=10*upper
            margin=H_lower-OL_upper
            assert ci!=0 and margin>0,(name,label,float(margin))
            row=dict(spectral_certificate=spec,waveform_exact=enc(v),
              squared_unit_norm_exact=str((v.T*v)[0]),OL_upper_increment=str(OL_upper),
              normal_MSE_exact=str(sum(s.trace(m['Pe'][r]) for r in ends)),
              trace_expected_Gram_exact=str(s.trace(R)),
              universal_causal_upper_formula='10*trR+2*sqrt(10*normal_MSE*trR); increment, conservative',
              witness_quadratic_increment=str(10*q),historical_c=enc(c),
              historical_Py=enc(py),Py_trace_upper_square=str(B*B),
              historical_cross_lower_exact=str(cross_lower),
              H_policy_increment_lower_exact=str(H_lower),
              strict_H_minus_OL_margin_exact=str(margin),
              strict_H_minus_OL_margin_display=float(margin),
              gain_source='actual-MSE cross term only; pointwise bias squared and hard coded energy unchanged')
            cs=iv.coefficients(m,times,ends,v,7)
            ell=sum(cs,s.zeros(1,2)).T
            C0=m['G'].T*iv.weights(m,7,ends)*iv.cov_ey(m,7,7)
            d=sum(cs[1:],s.zeros(1,2))
            row['current_integral_inputs']=dict(
              first_waveform=enc(v[:2,0]),C0=enc(C0),future_mean_row=enc(d),ell=enc(ell),
              Py=enc(m['Px'][7]+m['R'][0]),
              lambda_watermark='1/2',radial_mean='1',angle_period='pi')
            if label=='single':
                hcov=iv.history_genie_upper(m)
                row['history_genie_hcov']=enc(hcov)
                row['history_genie_hcov_trace_exact']=str(s.trace(hcov))
                row['H_upper_formula']='OL_upper_increment+2*sqrt(10*history_genie_hcov_trace_exact)'
            rows[label]=row
            print(json.dumps(dict(model=name,objective=label,strict_H_margin=float(margin))),flush=True)
        data[name]=rows
    result=dict(date='2026-11-17',models=data,
      exact_frozen_rational_inputs=True,physical_DARE_error_certified=False,
      positive_information_value_over_full_waveform_OL=True,
      no_physical_samples=True,global_causal_optimum_not_computed=True)
    (H/'exact_information_certificates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8')
if __name__=='__main__':main()
