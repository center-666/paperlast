"""Exact frozen-decimal checks; no model/gain redesign or physical samples."""
from pathlib import Path
from decimal import Decimal
from itertools import product
import json,hashlib
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent;src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
data=json.loads(src.read_text(encoding='utf-8'),parse_float=Decimal)
R=s.Rational
mat=lambda a:s.Matrix([[R(str(v))for v in row]for row in a])
Hs=list(map(mat,data['H']));Rs=list(map(mat,data['R']));Q=mat(data['Q'])
out=dict(date='2026-11-15',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 scope='exact frozen input K/G decimals, not a new end-to-end physical DARE certificate',models={})
for name,m in data['models'].items():
 A=mat(m['A']);K=[mat(r['K'])for r in m['nodes']];G=[mat(r['G'])for r in m['nodes']]
 def normal(i,S):
  C=s.eye(2)-K[i]*Hs[i]-sum((G[i]*Hs[j]for j in S),start=s.zeros(2));F=C*A
  V=C*Q*C.T+K[i]*Rs[i]*K[i].T+sum((G[i]*Rs[j]*G[i].T for j in S),start=s.zeros(2))
  a,b,c=s.symbols('p11 p12 p22');PP=s.Matrix([[a,b],[b,c]]);eq=PP-F*PP*F.T-V
  L,v=s.linear_eq_to_matrix([eq[0,0],eq[0,1],eq[1,1]],[a,b,c]);sol=L.inv()*v
  PP=PP.subs(dict(zip([a,b,c],sol)));assert PP-F*PP*F.T-V==s.zeros(2)
  assert PP[0,0]>0 and PP.det()>0
  return F,PP
 allbase=[normal(i,data['neighbors'][i])for i in range(5)]
 target=s.trace(allbase[1][1]);network=sum((s.trace(PP)for _,PP in allbase))/5
 F,PR=normal(1,[2]);tr=s.trace(PR);netR=network+(tr-target)/5
 assert tr>R(6,5)*target and netR>R(6,5)*network
 # Same original autonomous family; one preselected rational direction.
 if name=='unstable':xi=s.Matrix([R(-5372204,10**7),R(-2935868,10**7)]);packet=9
 else:xi=s.Matrix([R(-189,100),R(21,50)]);packet=0
 W=sum(((Hs[0]*A**t).T*(Hs[0]*A**t)for t in range(10)),start=s.zeros(2));energy=(xi.T*W*xi)[0]
 assert energy<10
 FA=allbase[1][0];terms=[FA**(packet-t)*G[1]*Hs[0]*A**t*xi for t in range(packet+1)]
 q=R(411,1000)if name=='unstable'else R(321,1000)
 contraction=[]
 for FF in [FA,F]:
  T=q*q*s.eye(2)-FF.T*FF;minor=[T[0,0],T.det()];assert all(v>0 for v in minor)
  contraction.append([str(v)for v in minor])
 gquarter=G[1].T*G[1]-R(1,16)*s.eye(2)
 gain_minor=[gquarter[0,0],gquarter.det()];assert all(v>0 for v in gain_minor)
 fupper=R(117,1000)if name=='unstable'else R(93,1000)
 TT=fupper*fupper*s.eye(2)-FA.T*FA
 first_norm_minor=[TT[0,0],TT.det()];assert all(v>0 for v in first_norm_minor)
 last_a=Hs[0]*A**packet*xi
 lower_input_sq=(last_a.T*last_a)[0]/16
 threshold_sq=(1+fupper)**2/R(10)if packet else R(1,10)
 assert lower_input_sq>threshold_sq
 if name=='unstable':
  safe_none=R(1,100)*W-Hs[0].T*G[1].T*G[1]*Hs[0]
  assert safe_none[0,0]>0 and safe_none.det()>0
  wg=W-13*s.eye(2);gg=R(74,1000)*s.eye(2)-G[1].T*G[1]
  assert all(v>0 for v in [wg[0,0],wg.det(),gg[0,0],gg.det()])
  # ||beta||<=1/2 is uniformly first-packet safe; its complement has
  # probability exp(-1/8) for beta~N(0,I2).
  safe_radius_squared_upper=R(10)*R(74,1000)/13*R(5,4)**2
  assert safe_radius_squared_upper<R(1,10)
  # For the same legal event witness, |beta dot unit(xi)|>=11/10 suffices
  # for first danger, independently of the perpendicular beta component.
  directional_lower_sq=(xi.T*xi)[0]/16*R(221,100)**2
  assert directional_lower_sq>R(1,10)
 damages=[]
 for signs in product([-1,1],repeat=packet+1):
  b=sum((v*sgn for v,sgn in zip(terms,signs)),start=s.zeros(2,1));damage=(b.T*b)[0]
  assert damage>R(1,10);damages.append(damage)
 out['models'][name]=dict(
  all_accept_target_MSE_exact=str(target),reject_target_MSE_exact=str(tr),
  target_ratio_exact=str(tr/target),target_ratio_decimal=str(s.N(tr/target,24)),
  all_accept_network_MSE_exact=str(network),reject_network_MSE_exact=str(netR),
  network_ratio_exact=str(netR/network),network_ratio_decimal=str(s.N(netR/network,24)),
  target_reject_exceeds_20pct=True,network_reject_exceeds_20pct=True,
  lyapunov_exact_residual_zero=True,
  all_target_admission_m_in_0_1_common_contraction=dict(
    q_exact=str(q),endpoint_norm_strict_upper_principal_minors=contraction,
    extension_to_fractional_m='convexity of induced2norm; F_m=(1-m)F_reject+mF_accept',
    establishes_uniform_homogeneous_contraction=True,does_not_establish_MSE_optimality=True),
  any_decoder_with_min_singular_value_at_least_one=dict(
    includes=['identity','whole sign','coordinate sign','orthogonal','Gaussian lambda1'],
    does_not_cover='Gaussian lambda<1 or scaled encoder without checking its own singular-value floor',
    gain_sigma_min_strict_lower='1/4',gain_certificate_minors=[str(v)for v in gain_minor],
    F_accept_norm_strict_upper=str(fupper),F_norm_certificate_minors=[str(v)for v in first_norm_minor],
    current_input_squared_norm_strict_lower=str(lower_input_sq),
    sufficient_critical_squared_norm=str(threshold_sq),
    strict_margin_exact=str(lower_input_sq-threshold_sq),
    danger_by_packet=packet,event_danger_probability_exact=1,
    proof='if all earlier bias norms <sqrtD, last b>=sigma_minG||M a||-||F||sqrtD>sqrtD; else event already happened'),
  sign_event_witness=dict(xi=[str(v)for v in xi],energy_exact=str(energy),energy_decimal=str(s.N(energy,24)),
    dangerous_packet=packet,state_index=packet+1,word_prefixes_checked=len(damages),
    minimum_squared_bias_exact=str(min(damages)),minimum_squared_bias_decimal=str(s.N(min(damages),24)),
    strict_gt_D_for_every_sign_prefix=True,full_L10_sign_words_covered=1024,
    same_direction_for_all_sign_words=True,event_danger_probability_exact=1))
 if name=='unstable':
  out['models'][name]['first_packet_probability_bounds']=dict(
    no_encoding_or_whole_sign_exact=0,
    no_encoding_safety_leading_minors=[str(safe_none[0,0]),str(safe_none.det())],
    Gaussian_lambda_q_1_worst_lower='2*(1-Phi(11/10))',Gaussian_lambda_q_1_worst_upper='exp(-1/8)',
    Gaussian_lower_uses_same_legal_event_witness=True,
    Gaussian_upper_uniform_all_energy_legal_xi=True,
    upper_safe_event_beta_radius='1/2',safe_event_squared_bias_upper=str(safe_radius_squared_upper),
    lower_directional_event_squared_bias_lower=str(directional_lower_sq),
    not_an_exact_global_probability_optimum=True)
 else:
  out['models'][name]['first_packet_probability_bounds']=dict(
    no_encoding_or_whole_sign_exact=1,Gaussian_lambda_q_1_worst_exact=1,
    witness='same legal fixed xi above; min singular decoder at least1')
(H/'fixed_gain_exact_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:dict(target_ratio=v['target_ratio_decimal'],network_ratio=v['network_ratio_decimal'],sign_min=v['sign_event_witness']['minimum_squared_bias_decimal'],energy=v['sign_event_witness']['energy_decimal'])for k,v in out['models'].items()}))
