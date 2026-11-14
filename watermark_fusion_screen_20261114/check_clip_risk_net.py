"""Uniform first-packet risk upper bound via Lipschitz covering and concentration.

Only this diagnostic uses random samples. Candidate is evaluated on fresh seeds.
The clipping upper event covers the CONTINUOUS energy ball, not just grid attacks.
Model parameters, stationary-error initialization approximation and MSE estimates
are reported separately; no general closed-loop worst risk theorem is claimed.
"""
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent
cfg=json.loads((H/'clip_risk_net_config.json').read_text())
data=json.loads((H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text())
clip=json.loads((H/'clip_baseline_results.json').read_text());base=clip['config']
normal=json.loads((H/'rho_normal_screen.json').read_text())
first=json.loads((H/'rho_first_packet_screen.json').read_text())
T=base['burn_steps']+base['evaluation_steps'];N=cfg['net_validation_paths'];delta=cfg['failure_probability_total']/4
noise=np.random.default_rng(cfg['net_validation_seed']).standard_normal((T,N,8))
initial=np.random.default_rng(cfg['net_initial_seed']).standard_normal((N,2))
beta=np.random.default_rng(cfg['net_watermark_seed']).standard_normal((N,2))
M=np.eye(2)[None,:,:]+beta[:,:,None]*beta[:,None,:]
# Side files differ in schema; select records by their explicit model/eta keys.
def find(d,model,eta,scheme=None):
 if isinstance(d,dict):
  if d.get('model')==model and abs(d.get('eta',-1)-eta)<1e-10 and (scheme is None or d.get('scheme')==scheme):return d
  for v in d.values():
   z=find(v,model,eta,scheme)
   if z is not None:return z
 elif isinstance(d,list):
  for v in d:
   z=find(v,model,eta,scheme)
   if z is not None:return z
 return None
def opnorm(A):
 tr=np.sum(A*A,axis=(-2,-1));det=np.linalg.det(A)**2
 return np.sqrt((tr+np.sqrt(np.maximum(tr*tr-4*det,0)))/2)
out=dict(config=cfg,rows=[],normal_precision_certified=False,global_variable_duration_risk_computed=False)
for name in cfg['models']:
 m=data['models'][name];nd=m['nodes'][1];A=np.array(m['A']);G=np.array(nd['G']);K=np.array(nd['K'])
 Hi=np.array(data['H'][1]);Ho=np.array(data['H'][2]);J0=np.eye(2)-K@Hi-G@Ho
 P0=np.array(nd['posterior_covariance']);row=find(clip['rows'],name,.2);tau=row['threshold_update_norm']
 e=initial@np.linalg.cholesky(P0).T;ref=e.copy();loss=np.zeros(N);lr=np.zeros(N)
 for t in range(T):
  pr=e@A.T+noise[t,:,:2];u=(pr+noise[t,:,6:]*np.sqrt(.5))@G.T
  c=u*np.minimum(1,tau/np.maximum(np.linalg.norm(u,axis=1),1e-100))[:,None]
  e=pr@J0.T-noise[t,:,2:4]@K.T*np.sqrt(.5)-noise[t,:,4:6]@G.T*np.sqrt(.5)-c
  pr0=ref@A.T+noise[t,:,:2];u0=(pr0+noise[t,:,6:]*np.sqrt(.5))@G.T
  ref=pr0@J0.T-noise[t,:,2:4]@K.T*np.sqrt(.5)-noise[t,:,4:6]@G.T*np.sqrt(.5)-u0
  if t>=base['burn_steps']:
   loss+=np.sum(e*e,axis=1);lr+=np.sum(ref*ref,axis=1)
 theta=np.zeros((8,8));R=.5*np.eye(2)
 for j in range(4):
  theta[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
  if j<3:
   theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T;theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
 inv=np.linalg.inv(theta);rs=np.concatenate([noise[t,:,:2]+noise[t,:,6:]*np.sqrt(.5)-noise[t-1,:,6:]@A.T*np.sqrt(.5)for t in range(T-4,T)],axis=1)
 q0=np.einsum('ni,ij,nj->n',rs,inv,rs);lin=(rs@inv)[:,-2:];W=inv[-2:,-2:]
 GM=np.einsum('ij,njk->nik',G,M);nw=opnorm(np.einsum('ij,njk->nik',np.linalg.cholesky(W).T,M));ng=opnorm(GM)
 Rmax=np.sqrt(10);na=cfg['angles_projective'];nr=cfg['radial_intervals']
 cover=Rmax/(2*nr)+2*Rmax*np.sin(np.pi/(4*na))
 aa=np.array([r*np.array([np.cos(th),np.sin(th)])for th in np.arange(na)*np.pi/na for r in np.linspace(0,Rmax,nr+1)])
 maxmean=-1.;witness=None;sth=np.sqrt(chi2.isf(.05/15,8))
 for off in range(0,len(aa),32):
  a=aa[off:off+32];eff=np.einsum('nij,bj->nbi',M,a);up=u[:,None,:]+np.einsum('ij,nbj->nbi',G,eff)
  up*=np.minimum(1,tau/np.maximum(np.linalg.norm(up,axis=2),1e-100))[:,:,None]
  bnorm=np.linalg.norm(up-c[:,None,:],axis=2)
  score=q0[:,None]+2*np.einsum('ni,nbi->nb',lin,eff)+np.einsum('nbi,ij,nbj->nb',eff,W,eff)
  indicator=(bnorm>=np.sqrt(.1)-ng[:,None]*cover)&(np.sqrt(np.maximum(score,0))<=sth+nw[:,None]*cover)
  means=indicator.mean(axis=0);ix=int(np.argmax(means))
  if means[ix]>maxmean:maxmean=float(means[ix]);witness=a[ix].tolist()
 half=float(np.sqrt(np.log(len(aa)/delta)/(2*N)))
 out['rows'].append(dict(model=name,eta=.2,cover_radius=cover,grid_points=len(aa),
    expanded_event_max_sample_mean=maxmean,Hoeffding_halfwidth=half,
    continuous_L1_risk_upper_qualified=min(1.,maxmean+half),grid_witness=witness,
    qualifier='model frozen; initialized finite-burn normal clipping trajectory, symmetric Gaussian law; not full variable-duration risk',
    normal_MSE_relative_increase_estimate=float(np.mean(loss-lr)/base['evaluation_steps']/np.trace(P0)),
    normal_MSE_relative_standard_error=float(np.std((loss-lr)/base['evaluation_steps'],ddof=1)/np.sqrt(N)/np.trace(P0)),
    fresh_seeds_after_initial_grid_screen=True))
 print(json.dumps(out['rows'][-1],ensure_ascii=False),flush=True)
(H/'clip_risk_net.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
