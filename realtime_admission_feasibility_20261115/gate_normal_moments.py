"""Exact-moment formulas for an existing spatial-consistency hard gate.

Floating matrix solves instantiate analytic Gaussian truncated moments. No
physical samples, new gate policy, parameter tuning or altered gains.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import chi2
from scipy.linalg import solve_discrete_lyapunov

H=Path(__file__).resolve().parent;P=H.parent
src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
d=json.loads(src.read_text(encoding='utf-8'))
Hs=[np.asarray(x,float)for x in d['H']];Rs=[np.asarray(x,float)for x in d['R']]
Q=np.asarray(d['Q'],float);I=np.eye(2)
alpha=0.05/15;tau=float(chi2.isf(alpha,2))
prob=1-alpha;cm=float(chi2.cdf(tau,4));diff=prob-cm
assert abs(diff+alpha*np.log(alpha))<1e-14
T=Hs[0]@np.linalg.inv(Hs[1]);S=Rs[0]+T@Rs[1]@T.T
Sinv=np.linalg.inv(S)
C00=prob*Rs[0]-diff*Rs[0]@Sinv@Rs[0]
C10=diff*Rs[1]@T.T@Sinv@Rs[0]
rows=[]
for name,m in d['models'].items():
 A=np.asarray(m['A'],float);node=m['nodes'][1]
 K=np.asarray(node['K'],float);G=np.asarray(node['G'],float)
 LA=I-K@Hs[1]-G@(Hs[0]+Hs[2]);LR=I-K@Hs[1]-G@Hs[2]
 FA=LA@A;FR=LR@A
 assert np.max(np.abs(FA-np.asarray(node['posterior_F'])))<1e-14
 N=K@Rs[1]@K.T+G@Rs[2]@G.T+G@C00@G.T+K@C10@G.T+G@C10.T@K.T
 Omega=prob*LA@Q@LA.T+(1-prob)*LR@Q@LR.T+N
 op=prob*np.kron(FA,FA)+(1-prob)*np.kron(FR,FR)
 rho=float(max(abs(np.linalg.eigvals(op))))
 assert rho<1
 cov=np.linalg.solve(np.eye(4)-op,Omega.reshape(-1,order='F')).reshape(2,2,order='F')
 cov=(cov+cov.T)/2
 residual=cov-(prob*FA@cov@FA.T+(1-prob)*FR@cov@FR.T+Omega)
 P0=np.asarray(node['posterior_covariance'],float)
 finite=[];Pk=P0.copy()
 for k in range(15):
  Pk=prob*FA@Pk@FA.T+(1-prob)*FR@Pk@FR.T+Omega
  finite.append(float(np.trace(Pk)))
 base=float(np.trace(P0));value=float(np.trace(cov))
 c_safe=float(np.sqrt(.1)*(1-np.linalg.norm(FR,2))/(2*np.linalg.norm(G,2)))
 c_first=float(np.sqrt(.1)/(2*np.linalg.norm(G,2)))
 reject_cov=solve_discrete_lyapunov(FR,LR@Q@LR.T+K@Rs[1]@K.T+G@Rs[2]@G.T)
 reject_mse=float(np.trace(reject_cov))
 crop_lower_at_safe=float(max(0,np.sqrt(reject_mse)-np.sqrt(.1)/2)**2)
 rows.append(dict(model=name,F_accept=FA.tolist(),F_reject=FR.tolist(),
  eigenvalues_accept=np.linalg.eigvals(FA).tolist(),eigenvalues_reject=np.linalg.eigvals(FR).tolist(),
  accept_norm2=float(np.linalg.norm(FA,2)),reject_norm2=float(np.linalg.norm(FR,2)),
  mean_square_operator_spectral_radius=rho,modewise_common_Euclidean_contraction=bool(max(np.linalg.norm(FA,2),np.linalg.norm(FR,2))<1),
  healthy_noise_covariance=N.tolist(),total_affine_covariance=Omega.tolist(),
  normal_stationary_covariance=cov.tolist(),normal_stationary_MSE=value,
  base_all_accept_MSE=base,normal_relative_MSE_increase=value/base-1,
  covariance_equation_residual_max=float(np.max(abs(residual))),
  finite_15_MSE_from_all_accept_steady_initial=finite,
  finite_15_average_from_all_accept_steady_initial=float(np.mean(finite)),
  finite_15_normal_relative_increase=float(np.mean(finite)/base-1),
  clip_existing_uniform_safe_radius_strict_upper=c_safe,
  clip_existing_first_packet_safe_radius_strict_upper=c_first,
  reject_normal_stationary_MSE=reject_mse,
  clip_normal_MSE_lower_for_all_c_up_to_uniform_safe_radius=crop_lower_at_safe,
  clip_normal_relative_lower_for_certificate_radius_range=crop_lower_at_safe/base-1,
  crop_bound_status='sufficient existing norm bound; failing bound is not actual risk failure'))
out=dict(date='2026-11-15',model_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 policy='existing spatial-consistency indicator gate; reject target whole innovation if normalized residual exceeds tau',
 source=0,target=1,alpha_single=alpha,tests_fixed=15,global_FA_union_upper=.05,
 residual_dimension=2,threshold=tau,probability_accept_normal=prob,
 truncated_chi4_mass=cm,probability_minus_truncated_mass=diff,
 normal_spatial_covariance=S.tolist(),weighted_v0_second_moment=C00.tolist(),
 weighted_v1_v0_cross_moment=C10.tolist(),rows=rows,
 no_physical_samples=True,no_new_policy=True,no_estimator_gain_change=True,
 normal_covariance_formula_analytic=True,finite_decimal_matrix_numbers_interval_certified=False,
 does_not_report_gate_actual_attack_risk=True,
 assumptions=['same-time decoded neighbor raw measurement and trusted local measurement',
 'Gaussian independent sensor noises; healthy exact synchronous decoding',
 'fresh sensor noises independent of previous estimate and process noise',
 'only target 0->1 gate; other healthy neighbor2 accepted',
 'no trusted source copy, true attack label or extra verification channel'])
(H/'gate_normal_moments.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({r['model']:{k:r[k]for k in ['mean_square_operator_spectral_radius','normal_stationary_MSE','normal_relative_MSE_increase','finite_15_normal_relative_increase','clip_existing_uniform_safe_radius_strict_upper','clip_existing_first_packet_safe_radius_strict_upper']}for r in rows},ensure_ascii=False))
