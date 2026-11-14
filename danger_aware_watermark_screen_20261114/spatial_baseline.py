"""Trusted-local spatial consistency: deterministic ordinary detector baseline.

Unchanged original-compatible aggregate estimator. No watermarker or new GLRT
is designed. The rank2 matched spatial test is a closed-form standard reference.
All methods separately use the same 15-test Bonferroni budget.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent
source=H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json'
data=json.loads(source.read_text(encoding='utf-8'))
alpha=1/300;E=10.;D=.1;L=10;T=15

def setup(name):
 m=data['models'][name];A=np.array(m['A']);node=m['nodes'][1]
 F=np.array(node['posterior_F']);G=np.array(node['G'])
 Hs=[np.diag([1,1]),np.diag([1.1,1.2]),np.diag([1.2,1.4]),np.diag([1.3,1.1]),np.diag([1.4,1.3])]
 J=Hs[0]@np.linalg.inv(Hs[1]);V=.5*np.eye(2)+.5*J@J.T;Vi=np.linalg.inv(V)
 O=np.stack([np.linalg.matrix_power(A,t)for t in range(L)])
 W=np.einsum('tji,tjk->ik',O,O);w,U=eigh(W);Wi=(U/np.sqrt(w))@U.T
 B=np.zeros((2,2));Ss=[]
 for k in range(T):
  B=F@B+(G@O[k]if k<L else 0)
  Ss.append(B.T@B)
 Ss=np.array(Ss)
 Qs=np.einsum('ia,tij,jb->tab',Wi,Ss,Wi)
 # Spatial full-five ordinary energy and rank2 matched statistic have same NCP.
 G5=sum(O[t].T@Vi@O[t]for t in range(5))
 G0=Vi
 return dict(A=A,F=F,G=G,V=V,Vi=Vi,O=O,W=W,Wi=Wi,Ss=Ss,Qs=Qs,G5=G5,G0=G0)

def min_direction(p,C,df,danger=True):
 threshold=chi2.isf(alpha,df)
 def at(theta):
  u=p['Wi']@np.array([np.cos(theta),np.sin(theta)])
  peak=max(float(u@S@u)for S in p['Ss'])
  energy=D/peak if danger else E
  if energy>E*(1+1e-11):return (float('inf'),None)
  xi=np.sqrt(energy)*u;lam=float(xi@C@xi)
  val=float(ncx2.sf(threshold,df,lam))
  return val,dict(theta=float(theta%np.pi),xi=xi.tolist(),encoded_energy=energy,ncp=lam,
    no_watermark_peak=float(max(xi@S@xi for S in p['Ss'])),
    no_watermark_terminal=float(xi@p['Ss'][9]@xi),Pd=val,df=df,alpha=alpha)
 grid=np.arange(1024)*np.pi/1024
 values=np.array([at(t)[0]for t in grid]);i=int(np.argmin(values))
 step=np.pi/1024
 opt=minimize_scalar(lambda z:at(z%np.pi)[0],bounds=(grid[i]-step,grid[i]+step),method='bounded',options={'xatol':1e-13})
 return at(opt.x%np.pi)[1]

def exact_noWM_first_alarm(p,xi):
 # Spatial noise n_k=v_j,k-J v_i,k is independent across k; hence this product
 # is correct. It would NOT be correct for temporally overlapping parity tests.
 surv=1.;cdf=[];pmf=[];tau=chi2.isf(alpha,2)
 for k in range(T):
  a=p['O'][k]@xi if k<L else np.zeros(2)
  pd=float(ncx2.sf(tau,2,float(a@p['Vi']@a)))
  pmf.append(surv*pd);surv*=1-pd;cdf.append(1-surv)
 b=np.zeros(2);td=None
 for k in range(T):
  b=p['F']@b+(p['G']@p['O'][k]@xi if k<L else 0)
  if td is None and b@b>=D*(1-1e-11):td=k
 return dict(first_alarm_CDF=cdf,first_alarm_PMF=pmf,no_alarm_probability=surv,
  first_danger_update_index=td,danger_before_alarm_probability=(1-cdf[td])if td is not None else 0.,
  includes_same_packet_alarm_before_fusion=True,independent_current_spatial_tests=True,
  threshold_tie_note='saved boundary xi uses >=D with 1e-11 display tolerance only; no certified event claim at a floating equality')

def main():
 out=dict(date='2026-11-14',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 E=E,D=D,L=L,bias_states=list(range(1,16)),per_test_alpha=alpha,finite_tests=15,
 total_FA_upper=.05,healthy_spatial_current_FA_exact=1-(1-alpha)**15,
 no_watermark_spatial_detector_uses_current_trusted_local=True,
 fixed_candidate_start=True,unknown_start_scan_solved=False,new_detector=False,models={})
 for name in data['models']:
  p=setup(name)
  d=dict(V_spatial=p['V'].tolist(),V_spatial_eigenvalues=eigh(p['V'],eigvals_only=True).tolist(),
   spatial_G5=p['G5'].tolist(),spatial_G_current=p['G0'].tolist(),W=p['W'].tolist(),
   normal_MSE_unchanged=True,deterministic_comparisons={})
  for label,C,df in [('current_spatial',p['G0'],2),('five_packet_spatial_energy',p['G5'],10),('standard_five_packet_spatial_matching',p['G5'],2)]:
   row=dict(common_danger_min=min_direction(p,C,df,True),E10_min=min_direction(p,C,df,False),
    mean_NCP_is_not_used_as_power=True)
   # Independent analytic optimum: in the two primary models the maximum
   # no-encoding S is respectively S10/S1 for every direction (certified in
   # danger_geometry). Min C subject xi'Sstar xi>=D is D*lambda_min(C,Sstar),
   # provided its generalized eigenvector meets the SAME W energy budget.
   Sstar=p['Ss'][9 if name=='unstable'else 0]
   lam,vec=eigh(C,Sstar);witness=np.sqrt(D)*vec[:,0]
   assert witness@p['W']@witness<E
   ncp=float(D*lam[0]);pd=float(ncx2.sf(chi2.isf(alpha,df),df,ncp))
   assert abs(pd-row['common_danger_min']['Pd'])<1e-10
   row['closed_form_common_danger_check']=dict(ncp=ncp,Pd=pd,
    initial_state=witness.tolist(),energy=float(witness@p['W']@witness),
    formula='D lambda_min(C,Sstar); eigenvector energy feasible; global Rayleigh optimum in frozen numerical matrices',
    physical_interval_power_certificate=False)
   d['deterministic_comparisons'][label]=row
  probes={}
  for label in ['current_spatial','five_packet_spatial_energy','standard_five_packet_spatial_matching']:
   # Fixed analytic probe directions, evaluated at E10 as well as danger boundary.
   z=d['deterministic_comparisons'][label]
   for key in ['common_danger_min','E10_min']:
    xi=np.array(z[key]['xi']);probes[label+'_'+key]=exact_noWM_first_alarm(p,xi)
  d['current_spatial_first_alarm_probes']=probes
  # Conditioning on an exogenous watermark path preserves independence of the
  # spatial noises across time; random-M integration must remain outside product.
  # Current-packet damage vs current spatial NCP: NCP >=gamma ||G delta||^2.
  gamma=eigh(p['Vi'],p['G'].T@p['G'],eigvals_only=True)[0]
  d['instantaneous_ncp_per_current_packet_bias_lower_diagnostic']=float(gamma)
  out['models'][name]=d
  print(name,{k:round(v['common_danger_min']['Pd'],6)for k,v in d['deterministic_comparisons'].items()})
 (H/'spatial_baseline.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
