"""Replay ONLY the registered old samples; never modify sibling research files."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys

sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent


def signature(p):
    import numpy as np
    if p.suffix == '.npz':
        with np.load(p) as d:
            return {k: dict(shape=list(d[k].shape), dtype=str(d[k].dtype),
                            sha256=hashlib.sha256(d[k].tobytes()).hexdigest())
                    for k in sorted(d.files)}
    if p.suffix == '.json':
        data = json.loads(p.read_text('utf8'))
        content = json.dumps(data, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode('utf8')
    else:
        content = p.read_text('utf8').replace('\r\n', '\n').encode('utf8')
    return hashlib.sha256(content).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--checks-only', action='store_true',
                   help='Keep saved training/validation; rerun analytic checks and reports')
    args = a.parse_args()
    expected = json.loads((H / 'replay_expected.json').read_text('utf8'))
    for name, sig in expected['inputs'].items():
        assert signature(H / name) == sig, ('Frozen input mismatch', name)
    frozen = {name: signature(H / name) for name in expected['inputs']}
    programs = []
    if not args.checks_only:
        programs += ['screen_attack_classes.py', 'strong_A0_and_time_profile_checks.py']
    programs += ['attacker_observer_bounds.py', 'exact_math_certificates.py',
                 'joint_law_boundary.py', 'independent_checks.py',
                 'build_results_reports.py', 'finalize_fulltext_reports.py']
    for name in programs:
        print('Replaying ' + name, flush=True)
        r = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(H / name)],
                           cwd=H, capture_output=True, text=True, encoding='utf8')
        if r.returncode:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit(r.returncode)
        print('  passed', flush=True)
    compared = []
    for name, sig in expected['outputs'].items():
        assert signature(H / name) == sig, ('Replay output differs', name)
        compared.append(name)
    for name, sig in frozen.items():
        assert signature(H / name) == sig, ('Sibling input was modified', name)
    out = dict(
        date='2026-11-17', mode='checks-only' if args.checks_only else 'full registered replay',
        frozen_input_count=len(frozen), output_count=len(compared),
        compared_outputs=compared, all_semantic_outputs_match=True,
        NPZ_comparison='all array names, shapes, dtypes and raw value bytes; not zip timestamps',
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'sympy')},
        new_physical_seeds=0, selection_changed=False, sibling_files_changed=False,
        original_50_node_reproduction=False, new_defense=False)
    (H / 'replay_checks.json').write_text(json.dumps(
        out, ensure_ascii=False, indent=2) + '\n', 'utf8')
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
