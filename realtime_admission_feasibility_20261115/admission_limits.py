"""Independent finite bounds for existing admission/clipping baselines.
No policy is optimized or introduced, and no physical/noise samples are drawn.
"""
from pathlib import Path
import json,hashlib,math
from decimal import Decimal
import sympy as s
from scipy.stats import chi2,ncx2
from scipy.integrate import quad
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;R=s.Rational;I=s.eye(2)
src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
d=json.loads(src.read_text(encoding='utf-8'),parse_float=Decimal)
mat=lambda a:s.Matrix([[R(str(x))for x in row]for row in a])
Hs=list(map(mat,d['H']));Rs=list(map(mat,d['R']));Q=mat(d['Q'])
v=d['models']['stable']['nodes'][1];A=mat(d['models']['stable']['A']);K=mat(v['K']);G=mat(v['G'])
LA=I-K*Hs[1]-G*(Hs[0]+Hs[2]);FA=LA*A
V=LA*Q*LA.T+K*Rs[1]*K.T+G*(Rs[0]+Rs[2])*G.T
fresh=G*(Hs[0]*Q*Hs[0].T+Rs[0])*G.T
oracle=(Q.inv()+sum((Hs[j].T*Rs[j].inv()*Hs[j]for j in [0,1,2]),start=s.zeros(2))).inv()
def pd(X):
 assert X==X.T
 ms=[X[:k,:k].det()for k in range(1,X.rows+1)];assert all(x>0 for x in ms)
 return [str(x)for x in ms]
proofs={}
proofs['fresh_lower']=pd(fresh-R('0.0944')*I)
proofs['fresh_upper']=pd(R('0.1084')*I-fresh)
proofs['FA_norm']=pd(R('0.0922')**2*I-FA.T*FA)
proofs['GHA_norm']=pd(R('0.231')**2*I-(G*Hs[0]*A).T*(G*Hs[0]*A))
assert s.trace(V)<R('.225') and s.trace(oracle)>R('.2225')
Pbase=mat(v['posterior_covariance']);assert s.trace(Pbase)<R('.2268')
B=R('1.2')*s.trace(Pbase);assert B<R('.2721')
prev=R('.2721')+R('.2268')/15;assert prev<R('.2873')
def exp_bounds(x,n=80):
 lo=sum((x**k/s.factorial(k)for k in range(n+1)),start=s.Integer(0))
 rem=x**(n+1)/s.factorial(n+1)/(1-x/R(n+2))
 assert x<R(n+2)
 return lo,lo+rem
# Coarse exact inequalities suffice; no CDF library is used for certificates.
assert exp_bounds(R('.53'))[1]<R('1.7')
assert exp_bounds(R(5),10)[0]>100
assert (2*R('.0944')+R('.1'))/R('1.7')>R('.169')
assert 2*R('.1084')*R('.01')*6<R('.0131')
assert R('.231')**2*R('.2873')<R('.0154')
t,u=R('.0131'),R('.0154')
assert R('.057')-t-u>0 and (R('.057')-t-u)**2>4*t*u
assert R('.169')-R('.057')==R('.112')
assert R('.0922')**2*R('.2873')+R('.225')-R('.2225')<R('.005')
# (sqrt(.112)-sqrt(.005))^2 > .0675, hence normal MSE>.29.
assert (R('.112')+R('.005')-R('.0675'))**2>4*R('.112')*R('.005')
assert R('.2225')+R('.0675')==R('.29')
xi=s.Matrix([R('-1.89'),R('.42')]);norm=(xi.T*xi)[0]
proofs['G_min_quarter']=pd(G.T*G-R(1,16)*I)
W=sum(((A**k).T*A**k for k in range(10)),start=s.zeros(2));energy=(xi.T*W*xi)[0]
assert energy<10 and norm/16>R('.1')
# Standard spatial gate fails the first-packet risk target already in As.
T=Hs[0]*Hs[1].inv();S=Rs[0]+T*Rs[1]*T.T;sl=min(S[0,0],S[1,1])
assert (R(5,4)**2*norm/sl)<7
assert exp_bounds(R('5.7'),100)[1]<300
assert (R('.7')+R('2.65'))**2<R('11.4') and R('2.65')**2>7
gate_risk_none=R(49,249);gate_risk_gauss=R(1,9)*gate_risk_none
assert gate_risk_gauss>R('.01')
# Uniform normal lower bound for any target innovation output with ||phi||<=c.
C=s.Matrix.vstack(Hs[1],Hs[0],Hs[2]);RR=s.diag(Rs[1],Rs[0],Rs[2]);SY=C*Q*C.T+RR
Lstar=Q*C.T*SY.inv();Lreject=s.Matrix.hstack(K,s.zeros(2),G)
Zcov=(Lstar-Lreject)*SY*(Lstar-Lreject).T
proofs['bounded_output_oracle_Z_lower']=pd(Zcov-R('.0816')*I)
proofs['G_norm269']=pd(R('.269')**2*I-G.T*G)
lam=R('.0816');r=R('.269')*R('.84');a=r*r/(2*lam)
assert 0<a<1
# Alternating Taylor lower bounds for exp(-a) and its finite integral.
N=15
el=sum(((-a)**k/s.factorial(k)for k in range(N+1)),start=s.Integer(0))
integ=sum(((-1)**k*r**(2*k+1)/((2*lam)**k*s.factorial(k)*(2*k+1))for k in range(N+1)),start=s.Integer(0))
xx=2*R(355,113)*lam;scale=10**30
sqrt_upper=R(math.isqrt(int(s.floor(xx*scale**2)))+1,scale)
assert sqrt_upper**2>xx
clip_mse_lower=R('.2225')+2*lam*el-r*sqrt_upper+2*r*integ
assert clip_mse_lower>R('.2724')>R('.2721')
# For all c>=.84, an ordinary uncoded first attack is dangerous on one ball.
c0=R('.84');rad=R('.24')
assert c0*c0<norm/4
assert 4*c0*c0*(xi.T*G.T*G*xi)[0]/norm>R('.449')**2
assert (R('.449')-2*R('.269')*rad)**2>R('.1')
U=A*Pbase*A.T+Q+Rs[0]
proofs['first_u_cov_min157']=pd(U-R('1.57')*I)
assert U.det()<R('1.59')**2 and norm/4<R('.969')**2
assert (R('.969')+rad)**2/(2*R('1.57'))<R('.47')
assert exp_bounds(R('.47'))[1]<R('1.61')
clip_first_risk_lower=rad**2/(2*R('1.59')*R('1.61'));assert clip_first_risk_lower>R('.01')
# Diagnostic bound for the Au Gaussian spatial gate, not a global certificate.
alpha=.05/15;tau=chi2.isf(alpha,2);au=d['models']['unstable'];AA=np.array(au['A'],float)
xxi=np.array([-.5372204,-.2935868]);rows=[]
for k in range(10):
 delta=np.linalg.matrix_power(AA,k)@xxi;l=float(delta@delta)/float(sl)
 p=max(0,float(ncx2.cdf(tau,2,l))-alpha)
 pg=quad(lambda b:max(0,float(ncx2.cdf(tau,2,(1+b)**2*l))-alpha)*math.exp(-b/2)/2,0,50,epsabs=1e-10)[0]
 rows.append(dict(k=k,none_lower=p,gaussian_lower_diagnostic=pg))
allmask_proofs={}
clip_example={}
for name,mm in d['models'].items():
 aa=mat(mm['A']);vv=mm['nodes'][1];kk=mat(vv['K']);gg=mat(vv['G'])
 allmask_proofs[name]={}
 for g0 in [0,1]:
  for g2 in [0,1]:
   ff=(I-kk*Hs[1]-gg*(g0*Hs[0]+g2*Hs[2]))*aa
   allmask_proofs[name][str((g0,g2))]=pd(R('.8')**2*I-ff.T*ff)
 fr=(I-kk*Hs[1]-gg*Hs[2])*aa
 fb=R('.411')if name=='unstable'else R('.321')
 gb=R('.271')if name=='unstable'else R('.269')
 pd(fb**2*I-fr.T*fr);pd(gb**2*I-gg.T*gg)
 assert (2*gb*R('.34')/(1-fb))**2<R('.1')
 frn=float(np.linalg.norm(np.array(fr,float),2));gn=float(np.linalg.norm(np.array(gg,float),2))
 rmax=gn*.34/(1-frn)
 clip_example[name]=dict(c=.34,full_event_bias_upper=2*rmax,
  risk_upper_zero_by_strict_norm_bound=2*rmax<math.sqrt(.1),
  rational_norms_certificate=[str(fb),str(gb)],rational_bias_upper=str(2*gb*R('.34')/(1-fb)))
out=dict(date='2026-11-15',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 proof_scope='exact rational frozen common K/G; not original50 nor physical uncertainty certificate',
 binary_target_only=dict(healthy_neighbor2_always_accepted=True,unknown_onset_uniform_risk_required=True,
  finite_normal_horizon=15,first_evaluated_packet_has_fresh_Q=True,normal_MSE_upper_assumed=str(B),
  normal_MSE_necessary_strict_lower='29/100',risk_target='1/100',
  verdict='no feasible target-only binary admission policy under these conditions',
  excludes=['joint admission on healthy neighbor2','clipping/saturation','trusted extra channels','known Ta only','only network-mean MSE constraint']),
 spatial_gate=dict(alpha=alpha,As_first_risk_lower_none=str(gate_risk_none),
  As_first_risk_lower_gaussian_lambda_q1=str(gate_risk_gauss),same_healthy_attacked_policy=True,
  Au_none_event_risk_lower_diagnostic=float(np.prod([r['none_lower']for r in rows])*(1-alpha)**5),
  Au_gaussian_event_risk_lower_diagnostic=float(np.prod([r['gaussian_lower_diagnostic']for r in rows])*(1-alpha)**5),
  Au_gaussian_failure_at_1percent_not_proven=True),
 ordinary_clip=dict(c_split='21/25',normal_MSE_lower_for_c_at_most_split=str(clip_mse_lower),
  normal_lower_decimal=float(clip_mse_lower),As_uncoded_first_risk_lower_for_all_c_at_least_split=str(clip_first_risk_lower),
  first_risk_lower_decimal=float(clip_first_risk_lower),
  verdict='As uncoded ordinary innovation ball clip has no radius satisfying target20percent and1percent in stated initialization',
  gaussian_all_radii_excluded=False,adaptive_saturation_excluded=False),
 clip_zero_risk_example=clip_example,
 arbitrary_target_two_neighbor_mask_common_norm_upper='.8',allmask_positive_minors=allmask_proofs,
 exact_positive_minors=proofs,legal_xi_energy=str(energy),new_algorithms=0,physical_samples=0,
 mathematical_referee='internal separate derivation for target-only inequality; no external peer review')
(H/'admission_limits.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items()if k not in ['exact_positive_minors','legal_xi_energy']},ensure_ascii=False))
