from pathlib import Path
import json,hashlib,subprocess,sys,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def sig(p):
    if p.suffix=='.json':b=json.dumps(json.loads(p.read_text('utf8')),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    else:b=p.read_text('utf8').replace('\r\n','\n').encode()
    return hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--check-saved',action='store_true');args=a.parse_args()
    exp=json.loads((H/'replay_expected.json').read_text('utf8'))
    for n,h in exp['inputs'].items():assert sig(H/n)==h,('input mismatch',n)
    if not args.check_saved:
        for name in ['gain_models.py','coupling_certificates.py','risk_bounds.py','attack_comparison.py',
                     'quadratic_information_counterexample.py','independent_checks.py','build_reports.py']:
            print('Replaying '+name,flush=True)
            r=subprocess.run([sys.executable,'-B','-X','utf8',str(H/name)],cwd=H,capture_output=True,encoding='utf8')
            if r.returncode:print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
            print('  passed',flush=True)
    for n,h in exp['outputs'].items():assert sig(H/n)==h,('output mismatch',n)
    for n,h in exp['inputs'].items():assert sig(H/n)==h,('sibling input changed',n)
    out=dict(date='2026-11-17',full_replay=not args.check_saved,all_match=True,
      frozen_inputs=len(exp['inputs']),matched_outputs=len(exp['outputs']),
      old_physical_inputs_reused=128,new_physical_samples=0,new_attack_parameter_grid=False)
    (H/'replay_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out))
if __name__=='__main__':main()
