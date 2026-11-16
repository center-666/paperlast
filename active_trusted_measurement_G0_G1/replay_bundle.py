"""Replay scientific results from sibling frozen code; no network/physical MC."""
from pathlib import Path
import sys,json,hashlib,subprocess
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical_sha(p):
    return hashlib.sha256(json.dumps(json.loads(p.read_text('utf8')),sort_keys=True,
        ensure_ascii=False,separators=(',',':')).encode('utf8')).hexdigest()
def main():
    files=['structural_certificates.json','batch_power_results.json','prior_baseline_results.json',
           'online_bounds_results.json','independent_checks.json','research_decision.json']
    reports=['03_可信量测噪声与检测功效.md','04_标准联合检测强基线.md','05_最坏攻击方向与资源约束.md']
    expected={n:canonical_sha(H/n) for n in files}
    expected.update({n:sha(H/n) for n in reports})
    arrays={}
    for n in ('batch_matrices.npz','prior_baseline_matrices.npz','online_kernels.npz'):
        with np.load(H/n) as a:arrays[n]={k:a[k].copy() for k in a.files}
    inputs=json.loads((H/'scientific_inputs.json').read_text('utf8'))['sibling_inputs']
    for f,expected_hash in inputs.items():assert sha(H.parent/f)==expected_hash,f
    for f in ('structural_certificates.py','batch_power_and_attacks.py','prior_baseline_check.py',
              'online_bounds.py','independent_checks.py','build_reports.py'):
        print('Replaying '+f,flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(H/f)],cwd=H,check=True)
    matches={}
    for f,old in expected.items():
        matches[f]=(canonical_sha(H/f) if f.endswith('.json') else sha(H/f))==old
        assert matches[f],f
    compared=0
    for f,old in arrays.items():
        with np.load(H/f) as actual:
            assert set(actual.files)==set(old),f
            for k,v in old.items():
                assert np.array_equal(actual[k],v),(f,k,float(np.max(np.abs(actual[k]-v))))
                compared+=1
    for f,expected_hash in inputs.items():assert sha(H.parent/f)==expected_hash,f
    out=dict(date='2026-11-16',JSON_semantically_and_generated_reports_byte_identical=matches,
       compared_arrays=compared,all_arrays_exactly_identical=True,
       sibling_frozen_inputs_unchanged=True,new_physical_noise_samples=0,
       conditional_noise_formulas_recomputed=True,unconditional_Gaussian_integral_certified=False,
       global_danger_risk_certified=False)
    (H/'public_reproduction.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8')
    print(json.dumps(out,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
