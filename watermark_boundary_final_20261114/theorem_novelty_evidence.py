"""Read-only provenance and exact elementary boundaries, not new watermark design.

No protected research script imported or executed. Local PDFs only read/hash;
all output remains in this independent boundary-audit directory.
"""
from pathlib import Path
import json,hashlib
import sympy as sp
from scipy.stats import chi2,ncx2
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
lib=ROOT/'参考文献/文献库.json'
data=json.loads(lib.read_text(encoding='utf-8-sig'))
rows={r['id']:r for r in data['records']}
evidence=[]
for rid in ['FPN04','INTRO06','INTRO05','FPN16','FPN06']:
 r=rows[rid];p=ROOT/'参考文献'/r['file']
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 evidence.append(dict(id=rid,citation_key=r['citation_key'],title=r['title'],doi=r['doi'],
   file=str(p.relative_to(ROOT)),sha256=r['sha256'],
   this_round_new_read=(rid=='FPN04'),
   this_round_new_pdf_pages=[5,6,7] if rid=='FPN04'else [],
   this_round_new_printed_pages='110—112'if rid=='FPN04'else '',
   this_round_new_visual_pages=[6]if rid=='FPN04'else [],
   prior_read_scope_reused=r['read_scope'],complete_all_proofs_verified=False))
S=sp.Rational;Rot=sp.Matrix([[0,-1],[1,0]])
models={'unstable':sp.Matrix([[S(108,100),S(5,100)],[S(1,100),S(105,100)]]),
        'stable':sp.Matrix([[S(80,100),S(5,100)],[S(1,100),S(90,100)]])}
checks=[]
for name,A in models.items():
 C=Rot*A-A*Rot
 assert C.det()!=0 and A.det()!=0 and Rot.T*Rot==sp.eye(2)
 checks.append(dict(model=name,public_fixed_rotation=[[str(v)for v in row]for row in Rot.tolist()],
    commutator=[[str(v)for v in row]for row in C.tolist()],det_commutator=str(C.det()),
    complete_internal_window_no_nonzero_autonomous_blind_direction=True,
    scope='H0=I; autonomous sequence chosen in encoded coordinates; nontrivial window has at least one adjacent difference',
    no_detection_power_or_danger_guarantee_claimed=True))
alpha=.05/15;threshold=float(chi2.isf(alpha,8))
px=float(ncx2.sf(threshold,8,20));py=.89*alpha+.11*float(ncx2.sf(threshold,8,200))
assert 22>20 and py<px
out=dict(date='2026-11-14',reference_library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
    library_mutated=False,new_references_registered=False,
    literature_scope='Miao newly targeted pages; four other sources reuse existing designated-page evidence, no all-paper peer audit',
    evidence=evidence,public_rotation_checks=checks,
    ncp_inference_counterexample=dict(df=8,alpha=alpha,threshold=threshold,
       X='constant 20',Y='0 with probability .89, 200 with probability .11',
       EX=20,EY=22,powerX=px,powerY=py,
       scope='abstract NCP law only; not a resource-matched comparison of actual watermark schemes'),
    original_manuscript_or_gain_or_attack_changed=False)
(H/'theorem_novelty_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(public_rotation_checks=checks,ncp_powerX=px,ncp_powerY=py),ensure_ascii=False,indent=2))
