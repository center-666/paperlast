"""Independent fixed-gain admission audit, no new KF or defense design.

Read the existing original-compatible aggregate instance. All calculations are
deterministic; no noise trajectories or Monte Carlo samples are generated.
"""
from pathlib import Path
from decimal import Decimal
import hashlib,json
import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_lyapunov,eigh
from itertools import product

HERE=Path(__file__).resolve().parent;P=HERE.parent
src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
raw=src.read_bytes();d=json.loads(raw,parse_float=Decimal)
def ar(x):return np.asarray(x,dtype=float)
def rat(x):return sp.Matrix([[sp.Rational(str(v))for v in row]for row in x])
H=[ar(x)for x in d['H']];R=[ar(x)for x in d['R']];Q=ar(d['Q']);N=5;n=2
out=dict(date='2026-11-15',source=str(src),source_sha256=hashlib.sha256(raw).hexdigest(),
 scheme='remove the entire target neighbor innovation; K/G unchanged',
 target_link='0->1',target=1,source_node=0,energy=10,danger_squared=.1,L=10,
 no_new_KF=True,no_physical_samples=True,models={})
for name,m in d['models'].items():
 A=ar(m['A']);K=[ar(x['K'])for x in m['nodes']];G=[ar(x['G'])for x in m['nodes']]
 Aex=rat(m['A']);H0ex=rat(d['H'][0]);Kex=rat(m['nodes'][1]['K']);Gex=rat(m['nodes'][1]['G'])
 rows={};full_FV={}
 for label,allowed in [('all_accept',[0,2]),('reject_target_keep_healthy',[2]),('target_local_only',[])]:
  accepted=[list(x)for x in d['neighbors']];accepted[1]=allowed
  Cs=[];Fs=[];Ls=[]
  for i in range(N):
   C=np.eye(n)-K[i]@H[i]-sum((G[i]@H[j]for j in accepted[i]),start=np.zeros((n,n)))
   Cs.append(C);Fs.append(C@A)
   L=[np.zeros((n,n))for _ in range(N)];L[i]=K[i]
   for j in accepted[i]:L[j]=G[i]
   Ls.append(L)
  FF=np.zeros((N*n,N*n));V=np.zeros_like(FF)
  for i in range(N):
   FF[i*n:(i+1)*n,i*n:(i+1)*n]=Fs[i]
   for j in range(N):
    V[i*n:(i+1)*n,j*n:(j+1)*n]=Cs[i]@Q@Cs[j].T+sum((Ls[i][s]@R[s]@Ls[j][s].T for s in range(N)),start=np.zeros((n,n)))
  PP=solve_discrete_lyapunov(FF,V)
  tr=[float(np.trace(PP[i*n:(i+1)*n,i*n:(i+1)*n]))for i in range(N)]
  assert np.max(np.abs(PP-FF@PP@FF.T-V))<1e-12
  scalarP=solve_discrete_lyapunov(Fs[1],V[2:4,2:4])
  assert np.max(np.abs(scalarP-PP[2:4,2:4]))<1e-12
  Cex=sp.eye(2)-Kex*rat(d['H'][1])-sum((Gex*rat(d['H'][j])for j in allowed),start=sp.zeros(2))
  Fex=Cex*Aex;t=sp.trace(Fex);det=Fex.det();jury=[1-det,1-t+det,1+t+det]
  assert all(v>0 for v in jury)
  rows[label]=dict(target_accepted_neighbors=allowed,target_C=Cs[1].tolist(),
    target_F=Fs[1].tolist(),target_F_eigenvalues=np.linalg.eigvals(Fs[1]).tolist(),
    target_spectral_radius=float(max(abs(np.linalg.eigvals(Fs[1])))),
    target_operator_norm=float(np.linalg.norm(Fs[1],2)),
    frozen_decimal_model_exact_Jury_positive=[str(x)for x in jury],
    target_normal_posterior_covariance=PP[2:4,2:4].tolist(),target_normal_MSE=tr[1],
    all_nodes_MSE=tr,network_mean_MSE=float(np.mean(tr)),
    full_shared_noise_covariance=V.tolist(),full_stationary_covariance=PP.tolist(),
    lyapunov_residual_max=float(np.max(np.abs(PP-FF@PP@FF.T-V))))
  full_FV[label]=(FF,V,PP)
 for label,r in rows.items():
  r['target_MSE_ratio_to_all_accept']=r['target_normal_MSE']/rows['all_accept']['target_normal_MSE']
  r['network_MSE_ratio_to_all_accept']=r['network_mean_MSE']/rows['all_accept']['network_mean_MSE']
  r['steady_target_20pct_reference_pass']=r['target_MSE_ratio_to_all_accept']<=1.2
  # Separately declared finite transient diagnostic: same healthy all-accept
  # stationary entry covariance, policy begins at packet0, then 15 updates.
  # This is not substituted for the predeclared steady-state normal criterion.
  FF,V,_=full_FV[label];PP=full_FV['all_accept'][2].copy();ts=[];ns=[]
  for k in range(15):
   PP=FF@PP@FF.T+V
   ts.append(float(np.trace(PP[2:4,2:4])))
   ns.append(float(np.trace(PP)/5))
  r['finite15_from_common_nominal_stationary_entry']=dict(
   target_MSE_each_update=ts,network_MSE_each_update=ns,
   target_average_MSE=float(np.mean(ts)),target_terminal_MSE=ts[-1],
   network_average_MSE=float(np.mean(ns)),network_terminal_MSE=ns[-1],
   target_average_ratio_to_all_accept=float(np.mean(ts))/rows['all_accept']['target_normal_MSE'],
   network_average_ratio_to_all_accept=float(np.mean(ns))/rows['all_accept']['network_mean_MSE'],
   target_final_covariance=PP[2:4,2:4].tolist())
 # The first incoming attacked packet has no bias memory; W uses all L10
 # coded-domain samples, never substitutes a single-packet energy ball.
 Wex=sum(((H0ex*Aex**t).T*(H0ex*Aex**t)for t in range(10)),start=sp.zeros(2))
 W=ar(Wex.tolist());C=G[1].T@G[1];HH=H[0]
 def first(matrix):
  vals,vec=eigh(HH.T@matrix@HH,W);xi=np.sqrt(10)*vec[:,-1]
  return dict(max_squared_first_bias=10*float(vals[-1]),xi=xi.tolist(),
    coded_total_L10_energy=float(xi@W@xi),xi_norm_squared=float(xi@xi))
 first_none=first(C);first_gauss_mean=first(5*C+np.trace(C)*np.eye(2))
 # A simple G-min lower bound controls every Gaussian M=I+beta beta^T,
 # because ||M xi||>=||xi||. Test the pre-existing stable witness only.
 witness=np.array([-1.89,.42]);wenergy=float(witness@W@witness)
 witness_lower=float(np.linalg.svd(G[1],compute_uv=False)[-1]**2*(witness@witness))
 # Original P1 finite-prefix harmful direction, no encoding. Duration remains10;
 # bias is propagated after stop through packet14. The sign check below uses
 # one preselected direction against all1024 words, not a per-word attacker.
 F=ar(rows['all_accept']['target_F']);B=np.zeros((2,2));peaks=[]
 for k in range(15):
  B=F@B+(G[1]@H[0]@np.linalg.matrix_power(A,k)if k<10 else 0)
  val,vec=eigh(B.T@B,W);u=np.sqrt(10)*vec[:,-1]
  peaks.append(dict(packet=k,state_index=k+1,max_squared_bias=10*float(val[-1]),xi=u.tolist()))
 p1=max(peaks,key=lambda r:r['max_squared_bias']);u=np.asarray(p1['xi']);
 # Radial inward adjustment prevents a floating reported10.000000000000002
 # from serving as an exact legal-energy certificate. The original maximizers
 # above remain numerical diagnostics; this witness is conservatively scaled.
 u*=.999999
 sign_peak=[];sign_terminal=[]
 for signs in product([-1,1],repeat=10):
  b=np.zeros(2);vals=[]
  for k in range(15):
   b=F@b+(G[1]@(signs[k]*(H[0]@np.linalg.matrix_power(A,k)@u))if k<10 else 0)
   vals.append(float(b@b))
  sign_peak.append(max(vals));sign_terminal.append(vals[-1])
 common_f=max(rows[x]['target_operator_norm']for x in ['all_accept','reject_target_keep_healthy'])
 common_c=max(np.linalg.norm(ar(rows[x]['target_C']),2)for x in ['all_accept','reject_target_keep_healthy'])
 # One possible conservative L2 noise bound, valid even if admission depends
 # on the current data; no independence from the admission decision is used.
 noise_rms=common_c*np.sqrt(np.trace(Q))+np.sqrt(np.trace(K[1]@R[1]@K[1].T+G[1]@R[2]@G[1].T))+np.linalg.norm(G[1],2)*np.sqrt(np.trace(R[0]))
 out['models'][name]=dict(A=A.tolist(),target_K=K[1].tolist(),target_G=G[1].tolist(),
  normal_cases=rows,W10_exact=[[str(x)for x in row]for row in Wex.tolist()],
  nominal_covariance_agrees_with_frozen_DARE_max=float(np.max(np.abs(ar(m['nodes'][1]['posterior_covariance'])-ar(rows['all_accept']['target_normal_posterior_covariance'])))),
  first_bias=dict(no_encoding=first_none,whole_sign_realization_same_as_none=first_none,
    Gaussian_lambda_q_1_expected_worst=first_gauss_mean,
    Gaussian_realization_supremum='infinity for every nonzero xi, since decoder gain unbounded',
    permanent_reject_target=0,
    stable_old_witness=dict(xi=witness.tolist(),coded_L10_energy=wenergy,
      energy_legal=wenergy<=10,uniform_Gaussian_and_whole_sign_squared_bias_lower=witness_lower,
      legal_witness_proves_first_danger_probability_one=wenergy<=10 and witness_lower>=.1)),
  no_encoding_event=dict(prefix_maxima=peaks,overall_worst=p1,
     worst_danger_probability=1 if p1['max_squared_bias']>=.1 else 0),
  whole_sign_same_fixed_direction_enum=dict(xi=u.tolist(),coded_total_L10_energy=float(u@W@u),
   word_count=1024,minimum_over_words_of_event_peak=float(min(sign_peak)),
   maximum_over_words_of_event_peak=float(max(sign_peak)),
   danger_word_count=sum(v>=.1 for v in sign_peak),
   actual_event_danger_probability_for_this_fixed_direction=sum(v>=.1 for v in sign_peak)/1024,
   represents_one_legal_preselected_direction=True,not_per_word_optimized_attack=True,
   diagnostic_only_no_interval_certification=True),
  arbitrary_target_admission_stability=dict(m_range=[0,1],healthy_neighbor2_always_used=True,
    common_Euclidean_contraction_upper=common_f,process_noise_coefficient_norm_upper=float(common_c),
    conservative_normal_noise_L2_upper=float(noise_rms),
    conservative_normal_asymptotic_MSE_upper=float((noise_rms/(1-common_f))**2),
    bound_does_not_certify_20pct_precision=True,current_data_admission_allowed=True),
  scientific_scope='frozen decimal input exact Jury; floating Lyapunov/eigenvalues; no new full physical DARE interval certification')
HERE.mkdir(exist_ok=True)
(HERE/'fixed_gain_numbers.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({name:{'cases':{label:{k:v[k]for k in ['target_normal_MSE','target_MSE_ratio_to_all_accept','network_mean_MSE','target_F_eigenvalues','target_operator_norm']}for label,v in m['normal_cases'].items()},'first_bias':m['first_bias']}for name,m in out['models'].items()},ensure_ascii=False,indent=2))
