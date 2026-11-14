from pathlib import Path
import json
import numpy as np
from scipy.linalg import eigh
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent
d=json.loads((H/'danger_detection_comparison.json').read_text(encoding='utf-8'))
sp=json.loads((H/'spatial_baseline.json').read_text(encoding='utf-8'))
out=dict(date='2026-11-14',new_optimization=False,maximum_expected_bias_definition='max_k sup_xi E||b_k||^2; not E max_k||b_k||^2',models={})
for name,m in d['models'].items():
 W=np.array(m['W']);rows={}
 for label,row in m['schemes'].items():
  Ss=np.array(row['prefix_expected_moment_matrices'])
  vals=[10*eigh(S,W,eigvals_only=True)[-1]for S in Ss];k=int(np.argmax(vals))
  a=row['probes'][-1]['danger_set_minimum']
  rows[label]=dict(danger_min_Pd=a['Pd'],danger_min_initial_state=a['initial_state'],
   danger_min_encoded_energy=a['energy'],danger_min_expected_peak=a['scheme_max_expected_bias_squared'],
   danger_min_peak_amplification=a['peak_expected_amplification'],
   worst_time_expected_bias_squared=float(vals[k]),worst_time_bias_index=k+1,
   full_energy_min_Pd=row['probes'][-1]['full_energy_minimum']['Pd'])
 sm=sp['models'][name];V=np.array(sm['V_spatial']);cond=float(np.linalg.cond(V))
 baseline=sm['deterministic_comparisons']['standard_five_packet_spatial_matching']['common_danger_min']
 # A fixed feasible xi gives an upper bound on each scheme's infimum. This
 # display uses a floating boundary witness and is a numerical diagnosis only.
 upper=float(ncx2.sf(chi2.isf(1/300,2),2,cond*baseline['ncp']))
 out['models'][name]=dict(schemes=rows,spatial=sm['deterministic_comparisons'],
   orthogonal_spatial_matching_ceiling_diagnostic=upper,
   ceiling_scope='any exogenous orthogonal M, ordinary known-start spatial rank2 test, fixed floating feasible witness; not a certified global optimum or all-detector bound')
(H/'comparison_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for name,m in out['models'].items():
 print(name)
 for label,row in m['schemes'].items():print(label,round(row['danger_min_Pd'],8),round(row['worst_time_expected_bias_squared'],8),row['worst_time_bias_index'])
