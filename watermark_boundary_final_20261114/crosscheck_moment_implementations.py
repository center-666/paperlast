"""Compare the independent rational reconstruction with the numerical branch."""
from pathlib import Path
import json,numpy as np,sympy as sp
H=Path(__file__).resolve().parent
a=json.loads((H/'watermark_moment_boundary.json').read_text(encoding='utf-8'))
b=json.loads((H/'baseline_comparison.json').read_text(encoding='utf-8'))
arr=lambda x:np.array([[float(sp.Rational(str(v)))for v in row]for row in x])
rows=[]
for name,m in a['models'].items():
 for kind,d in m['expected_terminal_damage'].items():
  n=b['models'][name]['schemes']['public_R90_supplement'if kind=='public_R90'else kind]
  eg=arr(m['expected_detector'][kind]['G']);ng=np.array(n['expected_ncp_matrix'])
  es=np.array(d['S']);ns=np.array(n['S']);ew=arr(m['W10']);nw=np.array(n['W'])
  errs=dict(expected_ncp_matrix=float(np.max(abs(eg-ng))),
        expected_damage_matrix=float(np.max(abs(es-ns))),energy_W=float(np.max(abs(ew-nw))))
  assert max(errs.values())<1e-11,(name,kind,errs)
  rows.append(dict(model=name,scheme=kind,absolute_differences=errs))
out=dict(date='2026-11-14',comparison='independent SymPy closed form and independent DARE vs numerical branch',
    models=2,scheme_models=len(rows),external_independent_review=False,rows=rows)
(H/'moment_implementation_crosscheck.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=len(rows),maximum_absolute_difference=max(max(r['absolute_differences'].values())for r in rows))))
