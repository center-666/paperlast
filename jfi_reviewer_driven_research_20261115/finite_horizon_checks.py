"""Independent finite-horizon first-packet checks, not a new detector or attack.

Only a fixed original-family witness is evaluated. No old entrypoint imported,
no stochastic samples and no algorithm or parameter search.
"""
import hashlib,json,pathlib,math
import numpy as np
import sympy as sp
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq
from scipy.special import roots_laguerre, roots_legendre
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=HERE.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'

def smat(x): return sp.Matrix([[sp.Rational(str(v)) for v in row] for row in x])
def rational_taylor_cdf_upper(x):
    # For 0<=x<=8/5 these exp(-t^2/2) series terms decrease on integration.
    # Odd truncation terms subtracted; even last index gives upper bound.
    assert 0 <= x <= sp.Rational(8,5)
    total=sum((-1)**n*x**(2*n+1)/(2**n*sp.factorial(n)*(2*n+1)) for n in range(13))
    # Classical Archimedean 223/71 < pi implies 1/sqrt(2*pi)<399/1000.
    assert 2*sp.Rational(223,71)*sp.Rational(399,1000)**2>1
    return sp.Rational(1,2)+sp.Rational(399,1000)*total

def gaussian_oracle(xi,alpha,n):
    # beta~N(0,I2): ||beta||^2=2x, x~Exp(1), orientation uniform.
    # Known true state and current M, so original honest sender noise is .5 I.
    # This is an oracle ROC diagnostic; it is not a proposed deployment rule.
    radial,wr=roots_laguerre(n)
    leg,wl=roots_legendre(n)
    angle=(leg+1)*np.pi/4
    weights=(wr[:,None]*wl[None,:]/2).ravel()
    rad2=2*radial[:,None]
    ncp=(2*np.dot(xi,xi)*(1+(2*rad2+rad2**2)*np.cos(angle)[None,:]**2)).ravel()
    root=np.sqrt(ncp)
    def fa(c): return np.dot(weights,ndtr((-c-ncp/2)/root))
    threshold=brentq(lambda c: fa(c)-alpha,-30,100,xtol=1e-12)
    pd=np.dot(weights,ndtr((-threshold+ncp/2)/root))
    return {'order':n,'log_likelihood_threshold':float(threshold),'false_alarm':float(fa(threshold)),
         'power':float(pd),'miss':float(1-pd),'rigorous_quadrature_error_certified':False}

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=pathlib.Path,default=HERE/'finite_horizon_checks.json')
    args=parser.parse_args()
    data=json.loads(SOURCE.read_text(encoding='utf-8')); rows=[]
    for name,mod in data['models'].items():
        A=smat(mod['A']); node=mod['nodes'][1]; G=smat(node['G']); K=smat(node['K'])
        H=[smat(h) for h in data['H']]
        F=(sp.eye(2)-K*H[1]-G*sum((H[j] for j in node['neighbors']),sp.zeros(2)))*A
        fn=sp.Rational(3,25) if name=='unstable' else sp.Rational(1,10)
        fncheck=fn**2*sp.eye(2)-F.T*F
        gncheck=G.T*G-sp.eye(2)/16
        assert fncheck[0,0]>0 and fncheck.det()>0
        assert gncheck[0,0]>0 and gncheck.det()>0
        W=sum(((H[0]*A**t).T*(H[0]*A**t) for t in range(10)),sp.zeros(2))
        witnesses=[]
        for tag,xi in [('previous_As_witness',sp.Matrix([-sp.Rational(189,100),sp.Rational(21,50)])),
                      ('simple_fixed_direction',sp.Matrix([sp.Rational(13,10),0]))]:
            energy=(xi.T*W*xi)[0]; norm=(xi.T*xi)[0]
            witnesses.append({'name':tag,'xi':[str(v) for v in xi],'energy':str(energy),
                'energy_float':float(energy),'within_E10':bool(energy<=10),
                'unit_or_Gaussian_lambda1_first_bias_squared_lower':str(norm/16),
                'strictly_dangerous_at_first_packet_D01':bool(norm/16>sp.Rational(1,10)),
                'orthogonal_oracle':[
                   {'alpha':alpha,'ncp':float(2*norm),'power':float(ndtr(math.sqrt(float(2*norm))-ndtri(1-alpha))),
                    'miss':float(ndtr(ndtri(1-alpha)-math.sqrt(float(2*norm))))}
                   for alpha in [1/300,.05]]})
        rows.append({'model':name,'exact_F_norm_upper':str(fn),'exact_G_sigma_min_lower':'1/4',
         'F_PSD_principal_minors':[str(fncheck[0,0]),str(fncheck.det())],
         'G_PSD_principal_minors':[str(gncheck[0,0]),str(gncheck.det())],
         'witnesses':witnesses})
    # Same As L10 witness from previous audit, arbitrary first-budget <= event .05.
    assert rows[1]['witnesses'][0]['within_E10']
    assert rows[1]['witnesses'][0]['strictly_dangerous_at_first_packet_D01']
    # sqrt(7.497)<2.74; Phi(1.6)<.95 and Phi(1.14)<.875 => miss>1/8.
    assert sp.Rational(7497,1000)<sp.Rational(137,50)**2
    upper16=rational_taylor_cdf_upper(sp.Rational(8,5))
    upper114=rational_taylor_cdf_upper(sp.Rational(57,50))
    assert upper16<sp.Rational(19,20) and upper114<sp.Rational(7,8)
    gauss=[]
    # Gaussian part is fixed-parameter oracle quadrature diagnosis, not theorem.
    xi=np.array([-1.89,.42])
    for alpha in [1/300,.05]:
        gauss.append({'alpha':alpha,'orders':[gaussian_oracle(xi,alpha,n) for n in [32,64,96]]})
    source_files=[SOURCE,HERE.parent.parent/'第一篇小论文/secure distributed safety.tex',
       HERE.parent/'sections/02_model_detector.tex',HERE.parent/'sections/04_watermark.tex']
    result={'scope':'fixed common-G original-compatible 5-node matrices; first test only; no clean sender current noise replica',
        'input_canonical_json_sha256':hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest(),
        'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.exists()},
        'unavailable_provenance_inputs':[str(p) for p in source_files if not p.exists()],
        'models':rows,'strict_As_first_packet_risk_lower_orthogonal_event_alpha005':{
          'value':'1/8','event_allocation_requirement':'first nominal alarm size <=.05; no prior selection/alarm conditioning',
          'Phi_16_rational_upper':str(upper16),'Phi_114_rational_upper':str(upper114),
          'strict':True,'novel_method_claim':False},
        'Gaussian_lambda1_q1_known_secret_oracle_diagnostic':gauss,
        'physical_samples':0,'new_attacks_or_detectors':0,
        'exact_DARE_or_engineering_certified':False}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'row_summary':[{r['model']:[(v['name'],v['energy_float'],v['within_E10']) for v in r['witnesses']]} for r in rows],
      'orthogonal_strict_bound':'1/8','Gaussian_oracle':gauss},ensure_ascii=False))

if __name__=='__main__':main()
