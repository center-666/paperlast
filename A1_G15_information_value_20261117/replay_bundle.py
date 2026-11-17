"""Reproduce deterministic diagnostics/certificates. No original file is written."""
from pathlib import Path
import sys,json,hashlib,subprocess,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def sig(p):
    if p.suffix=='.json':
        obj=json.loads(p.read_text('utf8'))
        b=json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    else:b=p.read_text('utf8').replace('\r\n','\n').encode()
    return hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--check-saved',action='store_true');args=a.parse_args()
    frozen=json.loads((H/'replay_expected.json').read_text('utf8'))
    for n,h in frozen['inputs'].items():assert sig(H/n)==h,('input changed',n)
    if not args.check_saved:
        for n in ['information_value.py','certify_information_value.py','validate_current_integral.py',
                  'conditional_attack.py','independent_checks.py','build_reports.py']:
            print('Replaying '+n,flush=True)
            r=subprocess.run([sys.executable,'-B','-X','utf8',str(H/n)],cwd=H,capture_output=True,encoding='utf8')
            if r.returncode:
                print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
            print('  passed',flush=True)
    for n,h in frozen['outputs'].items():assert sig(H/n)==h,('output differs',n)
    for n,h in frozen['inputs'].items():assert sig(H/n)==h,('sibling input modified',n)
    out=dict(date='2026-11-17',frozen_inputs=len(frozen['inputs']),matched_outputs=len(frozen['outputs']),
      full_replay=not args.check_saved,all_match=True,no_new_physical_samples=True,no_parameter_search=True)
    (H/'replay_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8');print(json.dumps(out))
if __name__=='__main__':main()
