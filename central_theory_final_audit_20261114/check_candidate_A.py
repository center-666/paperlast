"""Exact matrix audit of existing finite autonomous attack, not an attack design."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
import sympy as s
H=Path(__file__).resolve().parent;P=H.parent
src=P/'original_estimator_gate_20261114/five_node_aggregate_model.json'
d=json.loads(src.read_text(encoding='utf-8'),parse_float=Decimal)
out=dict(date='2026-11-14',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 attack='existing a_t=A^t xi for 0<=t<10, otherwise0; source H0=I2',
 scope='same complete five-packet parity basis; four healthy packets before onset',
 no_new_attack_algorithm=True,no_simulation=True,models={})
for name,m in d['models'].items():
 A=s.Matrix([[s.Rational(str(x))for x in row]for row in m['A']])
 Z=s.zeros(8,10)
 for j in range(4):Z[2*j:2*j+2,2*j:2*j+2]=-A;Z[2*j:2*j+2,2*j+2:2*j+4]=s.eye(2)
 O=s.Matrix.vstack(*(A**j for j in range(5)))
 assert Z.rank()==8 and Z*O==s.zeros(8,2)
 maps=[]
 for k in range(15):
  T=s.Matrix.vstack(*(A**t if 0<=t<10 else s.zeros(2)for t in range(k-4,k+1)))
  maps.append(Z*T)
 internal=s.Matrix.vstack(*maps[4:10]);whole=s.Matrix.vstack(*maps)
 assert internal==s.zeros(internal.rows,2)
 assert maps[0].rank()==2 and maps[10].rank()==2 and whole.rank()==2
 G=s.Matrix([[s.Rational(str(x))for x in row]for row in m['nodes'][1]['G']])
 assert G.det()!=0
 out['models'][name]=dict(A=[[str(x)for x in row]for row in A.tolist()],
  parity_rank=Z.rank(),observability_rank=O.rank(),ZO_exact_zero=True,
  internal_packet_ends=list(range(4,10)),internal_detection_rank=internal.rank(),
  start_packet0_rank=maps[0].rank(),first_stop_packet10_rank=maps[10].rank(),
  monitored_packet_ends=list(range(15)),whole_event_detection_rank=whole.rank(),
  internal_stealth_initial_space_dimension=2,whole_event_stealth_initial_space_dimension=0,
  nonzero_onset_all_event_strictly_stealthy_attack_exists=False,
  frozen_G_det=str(G.det()),per_window_ranks=[v.rank()for v in maps],
  first_stop_boundary_last_block=[str(x)for x in -A**10],
  exact_DARE_rank_bridge_is_existing_source=True)
(H/'candidate_A_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:{n:v[n]for n in ['internal_detection_rank','start_packet0_rank','whole_event_detection_rank']}for k,v in out['models'].items()}))
