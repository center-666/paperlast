"""Exact identities and conservative endpoint certificate for the rho screen.

No physical samples, parameter search, or old research script imports.
The prior exact DARE enclosure is used only to transfer a broad endpoint norm
margin from the frozen supplementary original-architecture model.
"""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
import json,hashlib
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent
source=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
cert_source=P/'original_estimator_gate_20261114/aggregate_gap_certificate.json'
d=json.loads(source.read_text(encoding='utf-8'),parse_float=Decimal)
cert=json.loads(cert_source.read_text(encoding='utf-8'))
R=lambda x:s.Rational(str(x))
mat=lambda z:s.Matrix([[R(x)for x in row]for row in z])
pd=lambda x:[s.factor(x[0,0]),s.factor(x.det())]
mu=s.Rational(3,5);var=s.Rational(1,25)
rho_pair=[s.Rational(2,5),s.Rational(4,5)]
testP=s.Matrix([[s.Rational(7,5),s.Rational(1,7)],[s.Rational(1,7),s.Rational(6,5)]])
rows=[]
for name,m in d['models'].items():
 A=mat(m['A']);n=m['nodes'][1]
 F1=mat(n['posterior_F']);G=mat(n['G']);K=mat(n['K'])
 Hi=mat(d['H'][1]);Hj=mat(d['H'][0]);Ho=mat(d['H'][2])
 I=s.eye(2);Q=I;Ri=Rj=Ro=I/2
 J0=I-K*Hi-G*Ho;L=G*Hj*A;F0=F1+L
 for F in [F0,F1]:
  minors=pd(s.Rational(419,1000)**2*I-F.T*F)
  assert all(v>0 for v in minors)
 # The separate DARE proof supplies <1e-10 error for F1/G. For the exact
 # F0=F1+G Hj A, ||Hj A||<2, so <3e-10. Endpoint .419 + error < .42.
 assert sum(x*x for x in Hj*A)<4
 previous=next(r for r in cert['rows']if r['model']==name)
 assert Fraction(previous['saved_F_G_operator_error_upper'])<Fraction(1,10**10)
 assert s.Rational(419,1000)+s.Rational(3,10**10)<s.Rational(42,100)
 def step(rho):
  J=J0-rho*G*Hj;F=J*A
  N=J*Q*J.T+K*Ri*K.T+G*Ro*G.T+rho*rho*G*Rj*G.T
  return F*testP*F.T+N
 left=sum((step(rho)for rho in rho_pair),start=s.zeros(2))/2
 right=step(mu)+var*(L*testP*L.T+G*(Hj*Q*Hj.T+Rj)*G.T)
 assert s.simplify(left-right)==s.zeros(2)
 rows.append(dict(model=name,
     frozen_endpoint_norm_bound='419/1000',exact_DARE_uniform_F_rho_norm_bound='21/50',
     exact_covariance_variance_identity=True,
     endpoint_positive_minors=[[str(v)for v in pd(s.Rational(419,1000)**2*I-F.T*F)]for F in [F0,F1]]))
gamma=s.Rational(9,100);E=10;c=s.Rational(21,50)
damage=gamma**2*E/(1-c*c)
assert damage<s.Rational(1,10)
# Scalar realization counterexample to a claim that the rho cap must keep
# randomized attack modulation in the fused update.
g=s.Rational(1,4);lam=1;gm=s.Rational(1,10)
scalar=[]
for M in [s.Integer(1),s.Integer(3),s.Integer(10)]:
 rho=gm/(g*M);assert rho<=1
 assert s.simplify(rho*g*M-gm)==0
 scalar.append(dict(M=str(M),rho=str(rho),effective_injection_gain=str(rho*g*M)))
out=dict(date='2026-11-14',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 exact_identity='two rational rho values verify the quadratic expansion; general proof in report',
 rows=rows,uniform_energy_gain=dict(gamma=str(gamma),E=E,c=str(c),
 all_time_squared_bias_upper=str(damage),strictly_below_D=True,
 scope='any finite coded-energy attack, zero entering shadow bias, same watermark-only rho in healthy/attack paths; fixed original K/G'),
 scalar_cancellation=scalar,physical_samples=0,new_algorithm=False)
(H/'rho_theory_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'uniform_damage_upper':str(damage),'floating_display':float(damage),
                  'endpoint_models':len(rows),'quadratic_covariance_identities_passed':True}))
