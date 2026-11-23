"""Mock only: validate end-to-end path-array and locked test data contracts.

This creates and removes synthetic fixtures in this folder. It is NOT a
physical experiment and does not modify any frozen paperlast source.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import run_crn_development as dev
import run_crn_final as fin
from selector_v3 import select
from test_adapter import fake_backend

HERE=Path(__file__).resolve().parent
FILES=['crn_development_losses.json','crn_development_manifest.json',
       'crn_locked_selection.json','crn_final_results.json','crn_final_path_arrays.npz']

def main():
    olddev,oldfin=dev.load_backend,fin.load_backend
    old_dev_here,old_fin_here=dev.HERE,fin.HERE
    with TemporaryDirectory(prefix='sscd_mock_final_') as tmp:
      workspace=Path(tmp)
      for source_name in ['watson_crn.py','selector_v3.py']:
          (workspace/source_name).write_bytes((HERE/source_name).read_bytes())
      dev.load_backend=fake_backend;fin.load_backend=fake_backend
      dev.HERE=workspace;fin.HERE=workspace
      try:
        sc={}
        for label in ('unstable','stable'):
            for angle in (0,-45,45):
                for onset in (50,65):
                    k=f'{label}__rot{angle:+.0f}__onset{onset}'
                    sc[k]={'path_ids':[f'fake_dev_{k}_{i}' for i in range(16)],
                           'losses':{name:[.06]*16 for name in dev.CANDIDATES}}
        data={'split':'development_crn','dimension':2,'theta':dev.CANDIDATES,
              'scenarios':sc,'source_reference':dev.EXPECTED_HEAD_REFERENCE}
        (workspace/'crn_development_losses.json').write_text(json.dumps(data)+'\n')
        lock=select(data)
        assert lock['selection']=='iid'
        (workspace/'crn_locked_selection.json').write_text(json.dumps(lock)+'\n')
        man={'seed':12345,'development_sha256':dev.sha(workspace/'crn_development_losses.json'),
             'lock_sha256':dev.sha(workspace/'crn_locked_selection.json'),
             'crn_generator_sha256':dev.sha(HERE/'watson_crn.py'),
             'selector_v3_sha256':dev.sha(HERE/'selector_v3.py'),
             'source_head_reference':dev.EXPECTED_HEAD_REFERENCE}
        (workspace/'crn_development_manifest.json').write_text(json.dumps(man)+'\n')
        fin.run(seed=54321,cal_paths=100,paths=16)
        out=json.loads((workspace/'crn_final_results.json').read_text())
        assert out['new_v3_selection']=='iid'
        assert len(out['models']['stable']['scenarios'])==6
        with np.load(workspace/'crn_final_path_arrays.npz') as dat:
            assert len(dat)>100 and all(v.shape[0] in (16,100) for v in dat.values())
        print('PASS synthetic final adapter: lock integrity, 12 scenarios, oracle, FWER, path archive; no real physics')
      finally:
        dev.load_backend=olddev;fin.load_backend=oldfin
        dev.HERE=old_dev_here;fin.HERE=old_fin_here

if __name__=='__main__':main()
