"""Finite-event risk validation with genuinely shared overlapping residual noise.

No estimator/noise physical trajectory is simulated: only the exact joint
Gaussian residual vector and deterministic attack-induced bias are evaluated.
Watermark Monte Carlo is explicit. Finite corpus selection is independent of
validation; the result is NOT a global worst-risk solution.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2,norm
from scipy.integrate import quad
H=Path(__file__).resolve().parent
cfg=json.loads((H/'event_risk_config.json').read_text())
data=json.loads((H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text())
SCHEMES=['Gaussian','normalized_Gaussian','whole_sign','coordinate_sign','Haar_SO2','identity','public_alternation','public_coordinate','public_R90']
nu=.5+.5*quad(lambda x:np.exp(-x)/(1+2*x)**2,0,np.inf)[0]
TAU=chi2.isf(.05/15,8);L=10;T=15
def watermarks(N,seed):
 rng=np.random.default_rng(seed);beta=rng.standard_normal((N,L,2))
 M=np.eye(2)[None,None,:,:]+beta[:,:,:,None]*beta[:,:,None,:]
 sg=2*rng.integers(0,2,(N,L))-1;cs=2*rng.integers(0,2,(N,L,2))-1
 ang=rng.uniform(0,2*np.pi,(N,L));R=np.empty((N,L,2,2))
 R[:,:,0,0]=R[:,:,1,1]=np.cos(ang);R[:,:,1,0]=np.sin(ang);R[:,:,0,1]=-np.sin(ang)
 eye=np.broadcast_to(np.eye(2),(N,L,2,2))
 co=np.ones((L,2));co[:,1]=(-1.)**np.arange(L)
 return dict(Gaussian=M,normalized_Gaussian=np.sqrt(nu)*M,
   whole_sign=sg[:,:,None,None]*np.eye(2),coordinate_sign=cs[:,:,:,None]*np.eye(2),
   Haar_SO2=R,identity=eye,public_alternation=eye*((-1.)**np.arange(L))[None,:,None,None],
   public_coordinate=eye*co[None,:,:,None],
   public_R90=np.broadcast_to(np.array([[0.,-1.],[1.,0.]]),(N,L,2,2)))
def problem(name,N,seed):
 m=data['models'][name];A=np.array(m['A']);node=m['nodes'][1]
 F=np.array(node['posterior_F']);G=np.array(node['G'])
 # q_l=y_l-A y_(l-1), l=-3..14; process and sensor noises kept correlated.
 nq=T+3;R=.5*np.eye(2);S=np.zeros((2*nq,2*nq))
 for j in range(nq):
  S[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
  if j+1<nq:
   S[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T;S[2*j+2:2*j+4,2*j:2*j+2]=-A@R
 Theta=S[:8,:8];inv=np.linalg.inv(Theta)
 qnoise=(np.random.default_rng(seed).standard_normal((N,2*nq))@np.linalg.cholesky(S).T).reshape(N,nq,2)
 # 15 windows are views of the SAME qnoise path, not independent noises.
 noise_windows=np.stack([qnoise[:,k:k+4].reshape(N,8)for k in range(T)],axis=1)
 O=np.stack([np.linalg.matrix_power(A,j)for j in range(L)])
 W=np.einsum('tji,tjk->ik',O,O)
 return dict(A=A,F=F,G=G,O=O,W=W,noise=noise_windows,inv=inv,joint_covariance=S)
def evaluate(p,M,xi,details=False):
 N=len(M);a=np.einsum('tij,j->ti',p['O'],xi);eff=np.einsum('ntij,tj->nti',M,a)
 decoded=np.zeros((N,T,2));decoded[:,:L]=eff
 qattack=np.zeros((N,T+3,2))
 # index j=3+k corresponds q_k; prior decoded input is zero for k=0.
 for k in range(T):
  prev=decoded[:,k-1]if k else np.zeros((N,2))
  qattack[:,3+k]=decoded[:,k]-prev@p['A'].T
 b=np.zeros((N,2));alive=np.ones(N,bool);failure=np.zeros(N,bool)
 danger_any=np.zeros(N,bool);alarms=np.zeros(N,bool);peak=np.zeros(N);terminal=None;first_danger=np.full(N,-1,int)
 for k in range(T):
  res=p['noise'][:,k]+qattack[:,k:k+4].reshape(N,8)
  score=np.einsum('ni,ij,nj->n',res,p['inv'],res)
  alarm=score>TAU;alarms|=alarm;alive&=~alarm
  b=b@p['F'].T+decoded[:,k]@p['G'].T
  sq=np.sum(b*b,axis=1);danger=sq>=.1
  first_danger[(first_danger<0)&danger]=k
  peak=np.maximum(peak,sq);danger_any|=danger;failure|=danger&alive
  if k==9:terminal=sq.copy()
 if not details:return float(failure.mean())
 return dict(risk=float(failure.mean()),danger_probability=float(danger_any.mean()),
     finite_horizon_detection_probability=float(alarms.mean()),
     expected_terminal_damage=float(terminal.mean()),mean_peak_damage=float(peak.mean()),
     first_danger_probabilities=[float(np.mean(first_danger==k))for k in range(T)])
def main():
 out=dict(date='2026-11-14',config=cfg,normal_estimator_update_unchanged=True,
   normal_mse_same_pathwise=True,global_worst_risk_solved=False,models={})
 selections={}
 for name in data['models']:
  tr=problem(name,cfg['training_paths'],cfg['training_noise_seed']);Ms=watermarks(cfg['training_paths'],cfg['training_watermark_seed'])
  ew,ev=eigh(tr['W']);Winv=ev@np.diag(1/np.sqrt(ew))@ev.T
  corpus=[]
  for theta in np.arange(cfg['attack_direction_count'])*np.pi/cfg['attack_direction_count']:
   u=Winv@np.array([np.cos(theta),np.sin(theta)])
   for frac in cfg['energy_fractions']:corpus.append(np.sqrt(10*frac)*u)
  corpus=np.array(corpus);selections[name]={}
  for scheme in SCHEMES:
   values=np.array([evaluate(tr,Ms[scheme],xi)for xi in corpus]);ix=np.argsort(values)[-cfg['selected_training_attacks_per_scheme']:][::-1]
   selections[name][scheme]=[dict(corpus_index=int(i),xi=corpus[i].tolist(),energy=float(corpus[i]@tr['W']@corpus[i]),training_risk=float(values[i]))for i in ix]
 # Lock selections before drawing ANY validation noise or watermark.
 (H/'event_risk_selection_locked.json').write_text(json.dumps(selections,indent=2)+'\n',encoding='utf-8')
 for name in data['models']:
  va=problem(name,cfg['validation_paths'],cfg['validation_noise_seed']);Ms=watermarks(cfg['validation_paths'],cfg['validation_watermark_seed'])
  results={}
  # All selected attacks form one common validation set for every scheme.
  common={tuple(a['xi'])for z in selections[name].values()for a in z};common=sorted(common)
  m=len(common)*len(SCHEMES)*len(data['models']);half=float(np.sqrt(np.log(m/.01)/(2*cfg['validation_paths'])))
  for scheme in SCHEMES:
   vals=[]
   for x in common:
    r=evaluate(va,Ms[scheme],np.array(x),True);r.update(xi=list(x),energy=float(np.array(x)@va['W']@np.array(x)));vals.append(r)
   best=max(vals,key=lambda r:r['risk'])
   results[scheme]=dict(validated_finite_common_corpus_max=best['risk'],
      lower_bound_full_family_worst_risk_confident=max(0.,best['risk']-half),
      one_sided_simultaneous_Hoeffding_halfwidth=half,best_attack=best,
      actual_global_worst_certified=False,all_fixed_attack_results=vals)
  healthy=np.einsum('nki,ij,nkj->nk',va['noise'],va['inv'],va['noise'])
  out['models'][name]=dict(common_corpus_count=len(common),schemes=results,
    healthy_event_false_alarm_empirical=float(np.mean(np.any(healthy>TAU,axis=1))),
    healthy_total_false_alarm_analytic_upper=.05,
    basic_difference_joint_covariance=va['joint_covariance'].tolist(),
    source_of_overlap='shared joint Gaussian q_l, windows reuse q rows',
    no_noise_window_independence_assumed=True)
  print(name,{k:round(v['validated_finite_common_corpus_max'],4)for k,v in results.items()},flush=True)
 (H/'event_risk_screen.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
