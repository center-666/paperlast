"""Read-only mathematical cross-checks; writes only this audit's evidence.

Does not run old certificate scripts, rebuild a DARE, simulate noise, or tune a
watermark. Exact finite calculations and a 70-digit scalar ROC diagnostic.
"""
from pathlib import Path
from decimal import Decimal
import hashlib, json
import sympy as s
import mpmath as mp

H=Path(__file__).resolve().parent; P=H.parent; R=s.Rational
names=['original_estimator_gate_20261114/five_node_aggregate_model.json',
       'original_estimator_gate_20261114/aggregate_gap_certificate.json',
       'innovation_positioning_final_20261114/resource_boundary_math.json',
       'watermark_boundary_final_20261114/unitary_early_boundary.json']
inputs={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in names}
model=json.loads((P/names[0]).read_text(encoding='utf-8'),parse_float=Decimal)
bridge=json.loads((P/names[1]).read_text(encoding='utf-8'))
A=s.Matrix([[R(str(v))for v in row]for row in model['models']['stable']['A']])
G=s.Matrix([[R(str(v))for v in row]for row in model['models']['stable']['nodes'][1]['G']])
xi=s.Matrix([R(-189,100),R(21,50)])
W=sum(((A**t).T*A**t for t in range(10)),start=s.zeros(2))
energy=(xi.T*W*xi)[0]; norm=(xi.T*xi)[0]
assert energy<10 and norm==R(7497,2000)
Q=G.T*G-R(501,2000)**2*s.eye(2)
minors=[Q[0,0],Q.det()]
assert all(v>0 for v in minors)
err=R(next(r for r in bridge['rows']if r['model']=='stable')['saved_F_G_operator_error_upper'])
assert R(501,2000)-err>R(1,4)
# This input is a prior physical error enclosure, not independently rebuilt here.
mp.mp.dps=70
nu=mp.mpf(3)/4-mp.exp(mp.mpf(1)/2)*mp.e1(mp.mpf(1)/2)/8
kappa=1/mp.sqrt(nu); d=mp.sqrt(mp.mpf('7.497')/nu)
z=mp.sqrt(2)*mp.erfinv(1-mp.mpf(2)/300)
miss=(1+mp.erf((z-d)/mp.sqrt(2)))/2
damage=mp.mpf(7497)/32000/nu
assert damage>mp.mpf('.1') and mp.mpf('.2345')<miss<mp.mpf('.2346')
# Correlated/singular Gaussian record: duplicate linear information is not
# independent. Exact support reduction agrees with the original coordinates.
L=s.Matrix([[1,0],[0,1],[1,1]]); K=L*L.T; v=s.Matrix([1,2]); u=L*v
true=(u.T*K.pinv()*u)[0]/2
marginal=sum((u[i]**2/K[i,i]/2 for i in range(3)),start=s.Integer(0))
assert true==R(5,2) and marginal==R(19,4)
out=dict(date='2026-11-14',baseline='c1dbba3',input_sha256=inputs,
 prior_physical_DARE_error_bridge_rederived_this_turn=False,
 C3=dict(quantifiers='inf policy sup original xi; same xi witness for all policy orientations',
  xi=[str(x)for x in xi],energy_exact=str(energy),energy_decimal=str(s.N(energy,30)),
  norm_squared_exact=str(norm),frozen_G_principal_minors=[str(x)for x in minors],
  previous_operator_error_upper=str(err),conditional_exact_G_sigma_min_lower='1/4',
  nu_exact='3/4-exp(1/2)*E1(1/2)/8',nu_diagnostic=str(nu),kappa_diagnostic=str(kappa),
  first_bias_squared_strict_lower_diagnostic=str(damage),
  oracle_shift_squared_exact='7.497/nu',first_size_exact='1/300',
  lower_risk_exact='Phi(Phi^(-1)(1-1/300)-sqrt(7.497/nu))',
  lower_risk_diagnostic=str(miss),probability_interval_certified=False,
  experiment_reduction='whiten and rotate oracle source innovation to N(0,I) versus N(d e1,I)',
  original_Gaussian_in_policy_class=False,all_finite_kappa_covered=False,
  independent_publication_center_established=False),
 B_singular_record_example=dict(L=[[str(x)for x in row]for row in L.tolist()],
  support_rank=K.rank(),exact_joint_KL=str(true),incorrect_sum_marginal_KL=str(marginal),
  original_physical_model_experiment=False),
 theorem_evidence=[
  dict(id='A',type='self-contained linear algebra proof plus independent exact finite check',
       direct_primary_theorem_comparison_complete=False,missing=['FPN07','FPN08','FPN11']),
  dict(id='B',type='conditional Gaussian/relative entropy identities under stated assumptions',
       joint_risk_representation=True,new_solved_global_risk_theorem=False),
  dict(id='C1',type='norm/positive trace equality proof; existing rational nu certificate retained'),
  dict(id='C2',type='Gaussian moment and change-of-variables consequence'),
  dict(id='C3',type='valid conditional class-uniform combination, standard oracle ROC reduction',
       value='nontrivial analysis insight; no established independent paper center')],
 sources=dict(FPN16_final='final_version_source.json',
  standard_NP_ROC='https://spseducation.tue.nl/disciplines/statistical/statisticalsignalprocessing_detection_hypothesis/',
  no_claim_of_full_field_novelty_clearance=True),
 no_new_attack_algorithm=True,no_new_watermark_or_detector=True,
 no_physical_samples=True,no_parameter_search=True,old_files_written=False,
 paper_decision='C: current independent publication center not established')
(H/'candidate_BC_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(energy=str(s.N(energy,14)),nu=str(nu)[:20],risk=str(miss)[:20],
 exact_joint_KL=str(true),incorrect_marginal_sum=str(marginal)),ensure_ascii=False))
