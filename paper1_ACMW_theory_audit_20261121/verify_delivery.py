"""Verify archived sources, links and working-tree preservation; no research rerun."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
git=['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]
manifest=json.loads((HERE/'input_manifest.json').read_text(encoding='utf-8'))
sources=[]
for entry in manifest['files']:
    data=(HERE/entry['path']).read_bytes()
    assert len(data)==entry['bytes'],entry['path']
    assert hashlib.sha256(data).hexdigest()==entry['sha256'],entry['path']
    sources.append(entry['path'])
for doc in (HERE/'README.md',HERE/'experiments.md',HERE/'dialogue.md',ROOT/'research_discussions/README.md'):
    for link in re.findall(r'\]\(([^()]+)\)',doc.read_text(encoding='utf-8')):
        if '://' not in link:
            assert (doc.parent/link.split('#')[0]).exists(),(doc,link)

# Verify frozen input bytes against the baseline commit, rather than merely
# trusting there was no edit to these files in this working session.
protected=['common_g_reference_revision_20261115/common_g_reference.py',
           'common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json',
           'B1_three_hypothesis_feasibility_20261120/registration.json']
for name in protected:
    old=subprocess.check_output(git+['show',manifest['baseline']+':'+name])
    current=(ROOT/name).read_bytes()
    # Tracked text files can use working-tree CRLF under the user's Git config.
    assert old.replace(b'\r\n',b'\n')==current.replace(b'\r\n',b'\n'),name
original=(HERE/'source_snapshots/BSCW/risk_frontier.py').read_bytes()
assert original==(HERE/'reproduction_BSCW_original/risk_frontier.py').read_bytes()
base=original.decode('utf-8-sig').replace('\r\n','\n').strip()
variant=(HERE/'reproduction_BSCW_corrected_noise/risk_frontier_corrected.py').read_text(encoding='utf-8-sig').replace('\r\n','\n').strip()
expected=base.replace('SIGMA = np.array([0.5, 0.5, 0.5, 0.58 / 1.12**2], dtype=float)',
                      'SIGMA = np.array([1.0, 0.5, 1.0 / 1.04**2, 0.58 / 1.12**2], dtype=float)')
expected=expected.replace('third variance fixed at 0.5; verify against frozen full model',
                         'Reviewer sensitivity: first/third variances replaced by the previously derived regional four-observation covariance; all other source settings retained, including the local parameter-norm energy constraint.')
assert variant==expected,'Unregistered BSCW sensitivity-code difference'

src=json.loads((HERE/'source_snapshots/BSCW/results.json').read_text())
rerun=json.loads((HERE/'reproduction_BSCW_original/results.json').read_text())
keys=['iid','bscw_a025','bscw_a030','bscw_a035','bscw_a040','bscw_a045','full_lc_tcw']
for row,key in zip(src['schemes'],keys):
    for source_field,new_field in [('fixed','risk_fixed'),('adaptive','risk_adaptive')]:
        assert abs(row[source_field]['mean']-rerun['schemes'][key][new_field]['mean'])<5.1e-7
for code in list(HERE.glob('*.py')):
    compile(code.read_text(encoding='utf-8-sig'),str(code),'exec')
result=dict(sources_verified=sources,protected_inputs=protected,links_verified=True,
            bscw_original_table_rounded_match=True,bscw_noise_variant_only_expected_changes=True,
            syntax_checked=True,python=sys.version.split()[0])
(HERE/'delivery_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
