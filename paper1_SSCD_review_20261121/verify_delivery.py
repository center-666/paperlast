"""Verify source preservation, numeric outputs and regression suite for archive."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
git=['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]
manifest=json.loads((HERE/'input_manifest.json').read_text())
for entry in manifest['files']:
    data=(HERE/entry['path']).read_bytes()
    assert len(data)==entry['bytes']
    assert hashlib.sha256(data).hexdigest()==entry['sha256'],entry['path']
for doc in (HERE/'README.md',HERE/'review.md',HERE/'experiments.md',HERE/'dialogue.md',ROOT/'research_discussions/README.md'):
    for link in re.findall(r'\]\(([^()]+)\)',doc.read_text(encoding='utf-8')):
        if '://' not in link:
            assert (doc.parent/link.split('#')[0]).exists(),(doc,link)
for name in ('common_g_reference_revision_20261115/common_g_reference.py',
             'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json',
             'B1_three_hypothesis_feasibility_20261120/registration.json',
             'paper1_ACMW_theory_audit_20261121/review_checks.py'):
    original=subprocess.check_output(git+['show',manifest['baseline']+':'+name])
    assert original.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),name
for path in HERE.glob('*.py'):compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
tests=subprocess.run([sys.executable,'-B',str(HERE/'test_sscd.py')],capture_output=True,text=True,encoding='utf-8')
(HERE/'test_log.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
assert tests.returncode==0,tests.stderr
r=json.loads((HERE/'results.json').read_text())
assert r['stats']['source_default_optimizer_failure']['error']=='ValueError'
assert all(f['error']=='ValueError' for f in r['stats']['source_endpoint_failures'])
assert abs(r['conflicting_family']['best']['zeta'])<1e-12
for row in r['fixtures']:
    d=row['design'];b=d['best'];iid=d['iid']
    assert b['feasible'] and d['converged']
    assert d['selected_policy_regret_upper']<.000501
    assert b['inner']['lower']>iid['inner']['upper']
    assert b['state_directionwise_change']<=1e-10
    assert abs(row['independent_detection']['full_attack_coded_energy']-10)<1e-9
h=json.loads((HERE/'heldout_results.json').read_text())
for row in h['fixtures']:
    import math
    for phi in row['direction_angles']:
        scaled=phi/(math.pi/256)
        assert abs(scaled-round(scaled))>.49
    for onset in row['onset_state_checks']:
        assert onset['directionwise_change']<=1e-10
    for time in row['pointwise_state_checks']:
        assert time['directional_change']<=1e-10
    for phase in row['phase_state_checks'].values():
        assert phase['selected_worst']<=phase['iid_worst']+1e-10
result=dict(source_bytes_verified=True,protected_previous_models_and_code_unchanged=True,
            document_links_verified=True,syntax_checked=True,regression_tests_passed=True,
            numerical_result_consistency_passed=True,
            analytic_bounds_are_outward_rounded=False,continuous_attack_parameter_dimension=2,
            original_50_node_validated=False,python=sys.version.split()[0])
(HERE/'delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
