"""Independent watermark-only sampling for one frozen candidate witness.

This validates a lower bound for the L1 worst risk, not a new attack search.
Conditional physical Gaussian miss probabilities are integrated analytically.
"""
from pathlib import Path
import json,numpy as np
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent
cfg=json.loads((H/'clip_risk_net_config.json').read_text())
model=json.loads((H.parent/'original_estimator_gate_20261114/five_node_aggregate_model.json').read_text())
normal=json.loads((H/'rho_normal_screen.json').read_text());first=json.loads((H/'rho_first_packet_screen.json').read_text())
N=cfg['candidate_independent_watermark_samples'];delta=cfg['failure_probability_total']/4
z=np.random.default_rng(cfg['candidate_seed']).standard_normal((N,2))
M=np.eye(2)[None,:,:]+z[:,:,None]*z[:,None,:];half=float(np.sqrt(np.log(1/delta)/(2*N)))
out=dict(date='2026-11-14',fresh_watermark_draws=N,physical_noise_draws=0,
         conditional_physical_miss='exact noncentral chi-square CDF for current window',rows=[])
for name in cfg['models']:
 A=np.array(model['models'][name]['A']);G=np.array(model['models'][name]['nodes'][1]['G'])
 nr=next(r for r in normal['models'][name]['rows']if r['eta']==.2)
 fr=next(r for r in first['models'][name]['rows']if r['eta']==.2)
 a=np.array(fr['Gaussian_aware']['attack']);gamma=nr['high_order']['gamma']
 GM=np.einsum('ij,njk->nik',G,M);tr=np.sum(GM*GM,axis=(1,2));det=np.linalg.det(GM)**2
 ng=np.sqrt((tr+np.sqrt(np.maximum(tr*tr-4*det,0)))/2);rho=np.minimum(1,gamma/ng)
 eff=np.einsum('nij,j->ni',M,a);b=rho[:,None]*(eff@G.T)
 theta=np.zeros((8,8));R=.5*np.eye(2)
 for j in range(4):
  theta[2*j:2*j+2,2*j:2*j+2]=np.eye(2)+R+A@R@A.T
  if j<3:
   theta[2*j:2*j+2,2*j+2:2*j+4]=-R@A.T;theta[2*j+2:2*j+4,2*j:2*j+2]=-A@R
 W=np.linalg.inv(theta)[-2:,-2:];lam=np.einsum('ni,ij,nj->n',eff,W,eff)
 vals=(np.sum(b*b,axis=1)>=.1)*ncx2.cdf(chi2.isf(.05/15,8),8,lam)
 mean=float(vals.mean());lower=max(0.,mean-half)
 out['rows'].append(dict(model=name,eta=.2,attack=a.tolist(),energy=float(a@a),gamma=gamma,
      witness_risk_mean=mean,one_sided_Hoeffding_halfwidth=half,
      L1_worst_risk_lower_qualified=lower,
      scope='specified preselected first-packet autonomous attack; frozen model and gamma; same detector/encoding',
      normal_precision_certified=False,full_variable_duration_worst_risk_computed=False))
(H/'candidate_first_mc.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
