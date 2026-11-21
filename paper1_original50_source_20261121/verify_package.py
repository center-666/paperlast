"""Source/config verification; never imports a legacy simulation module."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def verify():
    m=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    old_path=HERE.parent/'common_g_reference_revision_20261115/historical_source_evidence.json'
    historical=json.loads(old_path.read_text(encoding='utf-8')) if old_path.is_file() else None
    checked=[]
    for entry in m['files']:
        p=HERE/entry['path'];data=p.read_bytes()
        assert len(data)==entry['bytes']
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry['git_blob']
        ast.parse(data.decode('utf-8-sig'))
        if historical:
            candidates=[r for r in historical['code_records'] if r['file'].replace('\\','/').split('/')[-1]==p.name]
            assert any(r['sha256']==entry['sha256'] for r in candidates)
        checked.append(p.name)
    cfg=json.loads((HERE/'configuration_snapshot.json').read_text())
    W=np.array(cfg['graph']['adjacency']);H=np.array(cfg['H']);R=np.array(cfg['R'])
    assert W.shape==(50,50) and np.array_equal(W,W.T) and np.diag(W).sum()==0
    assert H.shape==(50,2,2) and R.shape==(50,2,2)
    assert not cfg['common_G_available'] and cfg['attack_links'][0]['physical']=='1->0'
    # Source has no extra local imports or input loaders; retain names in report.
    imports=set()
    for p in (HERE/'source').glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):imports.update(x.name.split('.')[0] for x in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
    assert imports<={'__future__','csv','os','warnings','pathlib','numpy','scipy','networkx','matplotlib'}
    parity=json.loads((HERE/'parity_verification.json').read_text())
    for model in ('unstable','stable'):
        assert parity[model]['max_abs_ZO']<1e-10 and parity[model]['min_theta_eigenvalue']>0
    with np.load(HERE/'parity_matrices_regenerated.npz',allow_pickle=False) as data:
        assert len(data.files)==200
    with np.load(HERE/'initial_stacked_gains_regenerated.npz',allow_pickle=False) as data:
        assert len(data.files)==100
    gain=json.loads((HERE/'gain_interface_audit.json').read_text())
    for model in ('unstable','stable'):assert gain[model]['common_G_equality_violations']>0
    plan=subprocess.run([sys.executable,'-B',str(HERE/'run_legacy.py')],capture_output=True,text=True,encoding='utf-8',check=True)
    assert not json.loads(plan.stdout)['execute']
    for p in HERE.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    result=dict(source_files=checked,sha256_and_git_blob_verified=True,
                matches_historical_hashes=bool(historical),configs_regenerated=True,
                graph_nodes=50,graph_edges=cfg['graph']['edge_count'],
                parity_arrays=200,initial_stack_gain_arrays=100,common_G_available=False,
                launcher_dry_run_verified=True,legacy_MC_executed=False,
                legacy_arbitrary_attack_optimization_executed=False)
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
