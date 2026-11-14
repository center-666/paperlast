"""Exact early-risk boundary for unitary single-packet codings, fixed L10.

No new attack mechanism: a_t=A^t xi with one published rational initial state.
Rational Theta/W and prior exact-DARE error enclosure are used. No Monte Carlo.
"""
from pathlib import Path
from decimal import Decimal
from math import factorial
import hashlib,json
import sympy as s
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent;P=H.parent;R=s.Rational
model=json.loads((P/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text(),parse_float=Decimal)
bridge=json.loads((P/'original_estimator_gate_20261114/aggregate_gap_certificate.json').read_text())
A=s.Matrix([[R(4,5),R(1,20)],[R(1,100),R(9,10)]]);I=s.eye(2)
W=sum(((A**t).T*A**t for t in range(10)),start=s.zeros(2))
xi=s.Matrix([R(-189,100),R(21,50)])
energy=(xi.T*W*xi)[0];normsq=(xi.T*xi)[0]
assert energy<10
G=s.Matrix([[R(str(v))for v in row]for row in model['models']['stable']['nodes'][1]['G']])
def pd(M):
 assert M==M.T
 minors=[M[:k,:k].det()for k in range(1,M.rows+1)]
 assert all(v>0 for v in minors)
 return [str(v)for v in minors]
# Frozen G has sigma_min>.2505; the exact DARE G differs by <1e-10.
gainproof=pd(G.T*G-R(501,2000)**2*I)
err=R(next(r for r in bridge['rows']if r['model']=='stable')['saved_F_G_operator_error_upper'])
assert err<R(1,10**10) and R(501,2000)-err>R(1,4)
damage_lower=normsq/16
assert damage_lower>R(1,10)
Theta=s.zeros(8)
for j in range(4):
 Theta[2*j:2*j+2,2*j:2*j+2]=I+I/2+A*A.T/2
 if j<3:
  Theta[2*j:2*j+2,2*j+2:2*j+4]=-A.T/2
  Theta[2*j+2:2*j+4,2*j:2*j+2]=-A/2
cur=Theta.inv()[-2:,-2:]
ncpproof=pd(R(3,5)*I-cur)
ncp_upper=R(3,5)*normsq
assert ncp_upper<R(9,4)
def exp_bounds(x,n=100):
 low=sum((x**k/s.Integer(factorial(k))for k in range(n+1)),start=s.Integer(0))
 term=x**(n+1)/s.Integer(factorial(n+1))
 upper=low+term/(1-x/R(n+2))
 assert x<R(n+2)
 return low,upper
# Central df8 tail at23 is >1/300, so the actual threshold is >23.
x=R(23,2);el,eu=exp_bounds(x)
poly=sum((x**k/s.Integer(factorial(k))for k in range(4)),start=s.Integer(0))
assert poly/eu>R(1,300)
# Noncentral df8 CDF at23, lambda9/4: positive Poisson mixture terms j0..12.
lamhalf=R(9,8);_,p_exp_upper=exp_bounds(lamhalf)
lower=s.Integer(0)
for j in range(13):
 central_lower=1-sum((x**k/s.Integer(factorial(k))for k in range(4+j)),start=s.Integer(0))/el
 if central_lower>0:lower+=lamhalf**j/s.Integer(factorial(j))/p_exp_upper*central_lower
assert lower>R(49,50)
out=dict(date='2026-11-14',scheme_class='any exogenous orthogonal decoder M_t, independent of physical noises, causal/correlated/public/secret allowed',
 input_sha256={n:hashlib.sha256((P/'original_estimator_gate_20261114'/n).read_bytes()).hexdigest()for n in ['five_node_aggregate_model.json','aggregate_gap_certificate.json']},
 assumptions=['b0=0; four healthy packets before attack0',
              'monitor starts at attack0; no conditioning on an earlier no-alarm record',
              'parameter sequence independent of all physical noises, no measurement-adaptive encoding',
              '15 fixed ordinary df8 parity tests, each alpha=1/300'],
 estimator='unchanged original-compatible aggregate target1 exact stabilizing DARE',
 model='stable A_s; fixed L10 autonomous a_t=A_s^t xi, energy<=10',
 xi=[str(v)for v in xi],actual_energy=str(energy),actual_energy_float=float(energy),
 xi_norm_squared=str(normsq),strict_uniform_first_bias_squared_lower=str(damage_lower),
 exact_DARE_G_sigma_min_lower='1/4',frozen_G_sigma_certificate=gainproof,
 current_detector_precision_upper='3/5',precision_certificate=ncpproof,
 current_ncp_upper=str(ncp_upper),detector_threshold_strict_lower=23,
 risk_lower_rational='49/50',exact_rational_CDF_lower=str(lower),
 conditional_miss_lower_float=float(lower),
 numerical_CDF_at_actual_threshold_and_9over4=float(ncx2.cdf(chi2.isf(1/300,8),8,2.25)),
 proof='uniform first danger; Lambda0<9/4; tau>23; monotonic noncentral CDF and finite positive Poisson-mixture lower bound; no window independence needed',
 first_alarm_timing='monitor starts at attack0; detect current packet before fusion; shadow bias until first alarm',
 full_event_risk_lower=True,new_defense_mechanism=False,new_attack_algorithm=False,
 limits=['specific compatible five-node stable model, D=.1/E10, ordinary parity/15-test budget',
         'not all invertible nonunitary encoders, not arbitrary detector no-go, not original 50-node certification',
         'not a claim of independent publication novelty'])
(H/'unitary_early_boundary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k]for k in ['xi','actual_energy_float','strict_uniform_first_bias_squared_lower','current_ncp_upper','risk_lower_rational','conditional_miss_lower_float']},indent=2))
