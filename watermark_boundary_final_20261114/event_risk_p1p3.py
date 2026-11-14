"""Evaluate saved P1/P3 directions, not a new adversarial optimizer.

The direction set comes exclusively from deterministic baseline comparison,
independent of event-validation draws. Risks are per fixed attack estimates.
"""
from pathlib import Path
import json,numpy as np
from event_risk_screen import H,problem,watermarks,evaluate,SCHEMES,cfg
source=json.loads((H/'baseline_comparison.json').read_text())
labels={'Gaussian':'Gaussian','normalized_Gaussian':'normalized_Gaussian','whole_sign':'whole_sign',
 'coordinate_sign':'coordinate_sign','Haar_SO2':'Haar_SO2','identity':'identity',
 'public_scalar_alternation':'public_alternation','public_coordinate_alternation':'public_coordinate',
 'public_R90_supplement':'public_R90'}
out=dict(date='2026-11-14',scope='fixed saved P1/P3 attacks; finite known-start event, no global worst-risk certificate',
         initial_states_source='baseline_comparison.json, deterministic direction/threshold responses',
         model_count=2,validation_paths=cfg['validation_paths'],models={})
for name,ms in source['models'].items():
 p=problem(name,cfg['validation_paths'],cfg['validation_noise_seed']);MM=watermarks(cfg['validation_paths'],cfg['validation_watermark_seed'])
 rows=[]
 for label,m in ms['schemes'].items():
  probe=m['probes'][-1]
  cases={'P1':{'initial_state':m['P1_initial_state']},'E10_min_power':probe['E10_min_power'],
         'P3_pot':probe['P3_pot'],'P3_phys':probe['P3_phys']}
  rr=dict(scheme=label,cases={})
  for kind,z in cases.items():
   if z.get('feasible') is False or z.get('initial_state') is None:
    rr['cases'][kind]=dict(feasible=False,risk=None,not_detection_100_percent=True);continue
   xi=np.array(z['initial_state']);r=evaluate(p,MM[labels[label]],xi,True)
   r.update(feasible=True,xi=xi.tolist(),energy=float(xi@p['W']@xi),
       fixed_attack_Hoeffding_halfwidth_99pct_simultaneous=float(np.sqrt(np.log(2*9*4/.01)/(2*cfg['validation_paths']))))
   rr['cases'][kind]=r
  rows.append(rr)
 out['models'][name]=rows
(H/'event_risk_p1p3.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print('Saved P1/P3 directions evaluated on shared correlated residual paths, not reselected on validation.')
