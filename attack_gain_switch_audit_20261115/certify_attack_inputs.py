"""Round feasibility toward the SAME E=10 before any sampling; no attack search."""
from pathlib import Path
import json,hashlib
import numpy as np
import sympy as sp
H=Path(__file__).resolve().parent;P=H.parent
def sm(x):return sp.Matrix([[sp.Rational(str(v)) for v in row] for row in x])
def main():
 assert not (H/'screen_results.json').exists()
 f=H/'analytic_preflight.json';a=json.loads(f.read_text('utf8'))
 if (H/'attack_feasibility_certificate.json').exists():raise RuntimeError('certificate already frozen')
 (H/'analytic_preflight_initial.json').write_bytes(f.read_bytes())
 source=json.loads((P/'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json').read_text('utf8'))
 cert=[]
 for name,z in a['models'].items():
  A=sm(source['models'][name]['A']);Hj=sm(source['H'][0]);W=sum(((Hj*A**t).T*(Hj*A**t) for t in range(10)),sp.zeros(2))
  for enc,row in z['attacks'].items():
   for key in ('ordinary','P1'):
    xi=np.array(row[key]['xi']);steps=0
    while True:
     v=sp.Matrix([sp.Rational(str(v)) for v in xi]);E=(v.T*W*v)[0]
     if E<=10:break
     xi=np.nextafter(xi,np.zeros(2));steps+=1;assert steps<100
    row[key]['xi']=xi.tolist();row[key]['actual_energy']=float(E)
    row[key]['exact_energy_rational']=str(E)
    cert.append({'model':name,'encoding':enc,'profile':key,'xi':xi.tolist(),'energy':str(E),'within_same_E10_exact':bool(E<=10),'toward_zero_ULP_steps':steps})
 a['attack_input_rounding']='minimal IEEE toward-zero rounding to satisfy exact rational energy<=10; no budget change or sample-dependent adjustment'
 f.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n','utf8')
 (H/'attack_feasibility_certificate.json').write_text(json.dumps({'cases':cert,'analytic_preflight_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'sampled_paths_at_certification':0},ensure_ascii=False,indent=2)+'\n','utf8')
 print(json.dumps({'energy_cases_certified':len(cert),'max_ULP_steps':max(x['toward_zero_ULP_steps'] for x in cert),'budget':10,'physical_samples':0}))
if __name__=='__main__':main()
