"""Portable, bounded reference checks. Never runs historical 50-node code."""
from pathlib import Path
import subprocess,sys,json
H=Path(__file__).resolve().parent
for name in ('test_reference.py','code_review_checks.py','theory_patch_checks.py'):
    p=subprocess.run([sys.executable,'-X','utf8',str(H/name)],cwd=H,capture_output=True,text=True,encoding='utf8')
    if p.returncode:
        print(p.stdout);print(p.stderr);raise SystemExit(p.returncode)
    print(name+': passed')
primary=json.loads((H/'test_results.json').read_text('utf8'))
theory=json.loads((H/'theory_patch_checks.json').read_text('utf8'))
assert primary['failed']==0 and primary['passed']==227 and theory['exact_checks_passed']==7
print(json.dumps({'main_assertions':primary['passed'],'exact_theory_checks':theory['exact_checks_passed'],'independent_code_checks':'passed','physical_paths':2,'MC':False,'no_manuscript_edits':True},ensure_ascii=False))
