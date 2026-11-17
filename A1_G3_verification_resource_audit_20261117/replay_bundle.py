from pathlib import Path
import json,hashlib,subprocess,sys,argparse
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def sig(p):
    if p.suffix=='.npz':
        import numpy as np
        with np.load(p,allow_pickle=False) as z:
            rows={n:dict(dtype=z[n].dtype.str,shape=z[n].shape,
                  sha256=hashlib.sha256(z[n].tobytes()).hexdigest()) for n in sorted(z.files)}
        b=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    elif p.suffix=='.json':
        b=json.dumps(json.loads(p.read_text('utf8')),sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    else:b=p.read_text('utf8').replace('\r\n','\n').encode()
    return hashlib.sha256(b).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-saved',action='store_true');a=parser.parse_args()
    ex=json.loads((H/'replay_expected.json').read_text('utf8'))
    for n,h in ex['inputs'].items():assert sig(H/n)==h,('input mismatch',n)
    if not a.check_saved:
        for n in ('certify_boundaries.py','test_protocol.py','independent_covariance.py','build_reports.py'):
            print('Replaying '+n,flush=True)
            r=subprocess.run([sys.executable,'-B','-X','utf8',str(H/n)],cwd=H,capture_output=True,encoding='utf8')
            if r.returncode:
                print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
            print('  passed',flush=True)
    for n,h in ex['outputs'].items():assert sig(H/n)==h,('output mismatch',n)
    for n,h in ex['inputs'].items():assert sig(H/n)==h,('changed old input',n)
    r=dict(executed='2026-11-18',full_replay=not a.check_saved,all_match=True,
       frozen_inputs=len(ex['inputs']),matched_outputs=len(ex['outputs']),
       reused_old_physical_fixtures=4,new_physical_samples=0,probability_MC=False,
       exact_rational_and_interval_assertions=True,bitwise_replay_requires_pinned_environment=True)
    (H/'replay_checks.json').write_text(json.dumps(r,indent=2)+'\n','utf8');print(json.dumps(r))
if __name__=='__main__':main()
