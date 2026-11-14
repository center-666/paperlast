"""Small target-node ordinary clipping reference, separate from rho candidate.

Actual Gaussian noise samples are used and reported as such. No claim of exact
stationary MSE or global worst risk. Other nodes only supply raw measurements,
so their estimates never enter this target's recursion.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.stats import chi2
H=Path(__file__).resolve().parent
cfg=json.loads((H/'clip_baseline_config.json').read_text())
data=json.loads((H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text())
T=cfg['burn_steps']+cfg['evaluation_steps'];Ntr=cfg['training_paths'];Nv=cfg['validation_paths']
def noises(seed,N):return np.random.default_rng(seed).standard_normal((T,N,8))
train=noises(cfg['training_seed'],Ntr);valid=noises(cfg['validation_seed'],Nv)
inittr=np.random.default_rng(cfg['training_seed']+1).standard_normal((Ntr,2))
initv=np.random.default_rng(cfg['validation_seed']+1).standard_normal((Nv,2))
beta=np.random.default_rng(cfg['watermark_seed']).standard_normal((Nv,2))
M=np.eye(2)[None,:,:]+beta[:,:,None]*beta[:,None,:]
out=dict(date='2026-11-14',config=cfg,noise_samples_generated=int((Ntr+Nv)*T*8),
         physical_estimator_trajectories_generated=True,Monte_Carlo=True,
         full_network_50_node_simulation=False,rows=[])
for name,m in data['models'].items():
 node=m['nodes'][1];A=np.array(m['A']);K=np.array(node['K']);G=np.array(node['G'])
 Hi=np.array(data['H'][1]);Hother=np.array(data['H'][2]);Hj=np.array(data['H'][0])
 J0=np.eye(2)-K@Hi-G@Hother
 P0=np.array(node['posterior_covariance']);trace=float(np.trace(P0))
 def run(tau,noise,initial):
  N=len(initial);e=initial@np.linalg.cholesky(P0).T;ref=e.copy();loss=np.zeros(N);refloss=np.zeros(N)
  for t in range(T):
   w=noise[t,:,:2];vi=noise[t,:,2:4]*np.sqrt(.5);vo=noise[t,:,4:6]*np.sqrt(.5);vj=noise[t,:,6:]*np.sqrt(.5)
   pred=e@A.T+w;u=(pred@Hj.T+vj)@G.T
   scale=np.minimum(1.,tau/np.maximum(np.linalg.norm(u,axis=1),1e-100))
   e=pred@J0.T-vi@K.T-vo@G.T-u*scale[:,None]
   pr=ref@A.T+w;ur=(pr@Hj.T+vj)@G.T
   ref=pr@J0.T-vi@K.T-vo@G.T-ur
   if t>=cfg['burn_steps']:
    loss+=np.sum(e*e,axis=1);refloss+=np.sum(ref*ref,axis=1)
  return loss/cfg['evaluation_steps'],refloss/cfg['evaluation_steps'],u
 # Same finite test family as the six linear/random-gain schemes.
 s=4;theta=np.zeros((8,8));R=.5*np.eye(2)
 for j in range(s):
  theta[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
  if j+1<s:
   theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T
   theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
 inv=np.linalg.inv(theta);rs=[]
 for t in range(T-4,T):rs.append(valid[t,:,:2]+valid[t,:,6:]*np.sqrt(.5)-valid[t-1,:,6:]@A.T*np.sqrt(.5))
 rs=np.concatenate(rs,axis=1);q0=np.einsum('bi,ij,bj->b',rs,inv,rs);lin=(rs@inv)[:,-2:];diag=inv[-2:,-2:]
 threshold=chi2.isf(cfg['alpha_current'],8)
 angles=np.arange(cfg['attack_angles'])*np.pi/cfg['attack_angles']
 radii=np.linspace(0,np.sqrt(cfg['E']),cfg['attack_radii'])
 attacks=np.array([r*np.array([np.cos(th),np.sin(th)])for th in angles for r in radii])
 for eta in cfg['budgets_eta']:
  lo,hi=0.,4.
  for _ in range(cfg['bisection_steps']):
   mid=(lo+hi)/2;ls,lr,_=run(mid,train,inittr)
   if np.mean(ls-lr)/trace>eta:lo=mid
   else:hi=mid
  tau=hi;ls,lr,u=run(tau,valid,initv);du=ls-lr
  clipped=u*np.minimum(1.,tau/np.maximum(np.linalg.norm(u,axis=1),1e-100))[:,None]
  risks={}
  for wm in ['none','Gaussian']:
   best=-1.;a_best=None
   for a in attacks:
    eff=np.broadcast_to(a,(Nv,2)) if wm=='none'else np.einsum('bij,j->bi',M,a)
    ua=u+eff@G.T
    up=ua*np.minimum(1.,tau/np.maximum(np.linalg.norm(ua,axis=1),1e-100))[:,None]
    damage=np.sum((up-clipped)**2,axis=1)
    score=q0+2*np.sum(lin*eff,axis=1)+np.einsum('bi,ij,bj->b',eff,diag,eff)
    risk=float(np.mean((damage>=cfg['D'])&(score<=threshold)))
    if risk>best:best=risk;a_best=a.tolist()
   bound=float(np.sqrt(np.log(2*len(attacks)/.05)/(2*Nv)))
   risks[wm]=dict(finite_attack_grid_empirical_max=best,attacker=a_best,
        simultaneous_95pct_Hoeffding_halfwidth=bound,
        global_worst_risk_computed=False)
  row=dict(model=name,eta=eta,threshold_update_norm=tau,
        actual_MSE_mean=float(ls.mean()),paired_relative_MSE_increase=float(du.mean()/trace),
        paired_relative_MSE_standard_error=float(du.std(ddof=1)/np.sqrt(Nv)/trace),
        healthy_reference_empirical_trace=float(lr.mean()),analytic_reference_trace=trace,
        empirical_single_test_false_alarm=float(np.mean(q0>threshold)),risks=risks,
        exact_same_normal_performance_for_none_Gaussian_sign=True,
        note='Clipping rule is fixed threshold of target update G*innovation, not fixed rho; budget and risks are Monte Carlo diagnostics, not certificates')
  out['rows'].append(row)
  print(json.dumps(row,ensure_ascii=False),flush=True)
(H/'clip_baseline_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
