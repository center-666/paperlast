"""Finite clock-translation checks; theorem is the algebraic shift identity.

Uses a deterministic legal path, not random Monte Carlo or a physical network.
"""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
P=HERE.parent
gph=json.loads((HERE/'graph_results.json').read_text(encoding='utf-8'))
inp=json.loads((P/'checks/水印收束_理论与映射核验_20261105.json').read_text(encoding='utf-8'))
F=np.array(inp['F']); g=np.array(inp['g']); W=np.array(inp['W10'])
edges={v:[] for v in gph['states']}
for row in gph['transitions']:edges[row['source_state']].append(row)
state=8
seq=[1 if state&(1<<(8-i)) else -1 for i in range(9)]
for k in range(200):
    options=edges[state]
    e=options[(k//3+k)%len(options)]
    seq.append(e['next_sign']); state=e['target_state']
safe=set(json.loads((HERE/'safe_words.json').read_text())['safe_indices'])
ids=[]
for t in range(len(seq)-9):
    w=sum(int(seq[t+j]>0)<<(9-j) for j in range(10));assert w in safe;ids.append(w)
adaptive=next(t for t in range(9,100) if seq[t-1]!=seq[t-2])
rows=[]
for Ta in [0,1,7,31,79,140,adaptive]:
    B=np.zeros((4,2)); absolute=np.zeros((4,2)); err=0.
    for clock in range(Ta+13):
        absolute=F@absolute
        if Ta<=clock<Ta+10:
            absolute+=np.outer(g,[1,(clock-Ta)/10])*seq[clock]
        if clock>=Ta:
            k=clock-Ta
            B=F@B+(np.outer(g,[1,k/10])*seq[clock] if k<10 else 0)
            err=max(err,float(np.max(np.abs(absolute-B))))
            assert np.linalg.eigvalsh(W/100-B.T@B)[0]>.00025-1e-14
    assert err<1e-14
    rows.append(dict(Ta=Ta,relative_absolute_map_difference=err,
                     history_adaptive_example=(Ta==adaptive)))
out=dict(date='2026-11-13',legal_deterministic_path=seq,all_window_indices=ids,
         translated_checks=rows,initial_bias_zero=True,clock_invariant_fixed_F_g=True,
         theorem_scope='single autonomous10-slot attack with all-time tail certificate; any stopping time',
         simulated_physical_trajectories=0,noise_samples=0,
         not_a_proof_by_testing=True)
(HERE/'unknown_onset_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print('checked all legal windows and seven onset translations; no physical/noise samples')
