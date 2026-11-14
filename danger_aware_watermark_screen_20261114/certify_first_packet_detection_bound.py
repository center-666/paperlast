"""Known-shift oracle NP bound, not a new implementable detector.

All rational inequalities apply to the fixed As model and original L10 attack.
The oracle gets true current state, xi and independent side information, but
not a clean copy/descendant of the attacked sender's current measurement noise.
"""
from pathlib import Path
from decimal import Decimal
from math import factorial
import hashlib,json
import sympy as s
from scipy.stats import norm,chi2,ncx2
H=Path(__file__).resolve().parent;P=H.parent;R=s.Rational
src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
cert=P/'original_estimator_gate_20261114/aggregate_gap_certificate.json'
m=json.loads(src.read_text(encoding='utf-8'),parse_float=Decimal)
bridge=json.loads(cert.read_text(encoding='utf-8'))
A=s.Matrix([[R(4,5),R(1,20)],[R(1,100),R(9,10)]])
xi=s.Matrix([R(-189,100),R(21,50)])
W=sum(((A**t).T*A**t for t in range(10)),start=s.zeros(2))
energy=(xi.T*W*xi)[0];normsq=(xi.T*xi)[0]
assert energy<10
G=s.Matrix([[R(str(v))for v in row]for row in m['models']['stable']['nodes'][1]['G']])
def pd(X):
 assert X==X.T
 v=[X[:j,:j].det()for j in range(1,X.rows+1)]
 assert all(x>0 for x in v)
 return [str(x)for x in v]
gaincert=pd(G.T*G-R(501,2000)**2*s.eye(2))
err=R(next(v for v in bridge['rows']if v['model']=='stable')['saved_F_G_operator_error_upper'])
assert R(501,2000)-err>R(1,4)
assert normsq/16>R(1,10)
oracle_ncp=2*normsq
assert oracle_ncp<R(137,50)**2 # sqrt(lambda)<2.74.
def exp_upper(x,n=100):
 lo=sum((x**j/s.Integer(factorial(j))for j in range(n+1)),start=s.Integer(0))
 return lo+x**(n+1)/s.Integer(factorial(n+1))/(1-x/R(n+2))
# pi<22/7 implies sqrt(2*pi)<251/100, hence phi(2.7)>
# (100/251)/exp_upper(729/200). Mills' strict lower bound Q(x)>phi(x)x/(1+x^2).
x=R(27,10)
mills_lower=R(100,251)/exp_upper(x*x/2)*x/(1+x*x)
assert mills_lower>R(1,300)
# Therefore z_(1-alpha)>2.7 and z-sqrt(lambda)>-.04.
# Phi(-.04)>.5-.04*.5=.48 since phi(t)<.5 for all t.
assert R(27,10)-R(137,50)==-R(1,25)
miss_lower=R(1,2)-R(1,25)*R(1,2)
assert miss_lower==R(12,25)
alpha=1/300
out=dict(date='2026-11-14',type='strict first-packet bound, not a new defense',
 input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [src,cert]},
 model='stable A_s, original-compatible common-G target1, b0=0',
 attack='a_t=A_s^t xi, t0..9; fixed before secret realization',
 xi=[str(x)for x in xi],energy=str(energy),energy_float=float(energy),xi_norm_squared=str(normsq),
 sigma_min_G_strict_lower='1/4',gain_principal_minor_certificate=gaincert,
 all_orthogonal_first_bias_squared_strict_lower=str(normsq/16),
 source_current_noise_covariance='(1/2)I_2',oracle_ncp=str(oracle_ncp),
 first_packet_alpha='1/300',oracle_power_numerical=float(norm.cdf((float(oracle_ncp))**.5-norm.isf(alpha))),
 oracle_miss_numerical=float(norm.cdf(norm.isf(alpha)-(float(oracle_ncp))**.5)),
 Mills_tail_lower_at_27over10=str(mills_lower),Mills_tail_lower_float=float(mills_lower),
 strict_risk_lower='12/25',strict_risk_lower_float=float(miss_lower),
 proof='NP lemma for known Gaussian shift; constant signal norm under any observed exogenous orthogonal M; Mills bound plus Gaussian density bound; first danger occurs before any later evidence',
 constraints=['monitor starts at attack0; first test has unconditional nominal size <=1/300',
 'orthogonal parameter sequence independent of all physical noises',
 'oracle side information is hypothesis-independent and independent of attacked sender current noise; true current state and all past allowed',
 'no clean current sender measurement, safe duplicate or updated-state descendant that reveals its current noise',
 'no data-dependent attack start, no prior alarm conditioning',
 'not a bound for arbitrary nonunitary encoders, other size allocations or all physical systems'],
 no_window_independence_needed=True,new_watermark_algorithm=False,new_attack_mechanism=False)
(H/'first_packet_detection_bound.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k]for k in ['energy_float','oracle_ncp','oracle_power_numerical','oracle_miss_numerical','Mills_tail_lower_float','strict_risk_lower']},indent=2))
