"""Mocked backend smoke validates runtime interface but proves no 50-node physics."""
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import numpy as np
import run_crn_development as mod

HERE=Path(__file__).resolve().parent

def fake_physical(model,count,seed):
    rg=np.random.default_rng(seed)
    return SimpleNamespace(x=rg.normal(size=(count,150,2)),
                           y=rg.normal(size=(count,150,5,2)))

def fake_world(model,physical,delta=None,gate_on=False,threshold=1.,gate_override=None):
    B=len(physical.x)
    if delta is None:delta=np.zeros((B,150,6,2))
    raw=np.sum(delta**2,axis=-1)
    scores=raw.copy()
    gates=scores>threshold if gate_on else np.zeros_like(scores,dtype=bool)
    if gate_override is not None:gates=np.asarray(gate_override,bool)
    post=physical.x[:,:,None,:]+.03+delta.mean(axis=2)[:,:,None,:]*.02
    post=np.broadcast_to(post,(B,150,5,2)).copy()
    return SimpleNamespace(post=post,scores=scores,gates=gates)

def fake_backend():
    attack=np.zeros((150,6,2));attack[50:110]=np.array([.5,0.1])
    model=SimpleNamespace(T=150,links=[(i,(i+1)%5) for i in range(6)],attack=attack,
                          receiver_ids=np.arange(5))
    old=SimpleNamespace(validate_fixtures=lambda:{'mock':'not a real checksum'},
                        rotated_raw_attack=lambda m,a:(m.attack.copy(),1.0))
    return old, lambda label:model, fake_physical, fake_world,lambda scores,p:{'threshold':1.0}

def main():
    old,old_here=mod.load_backend,mod.HERE
    with TemporaryDirectory(prefix='sscd_mock_dev_') as tmp:
        workspace=Path(tmp)
        for source_name in ['watson_crn.py','selector_v3.py']:
            (workspace/source_name).write_bytes((HERE/source_name).read_bytes())
        mod.load_backend=fake_backend;mod.HERE=workspace
        try:
            mod.run(seed=93244,cal_paths=100,dev_paths=16,smoke=True,allow_overwrite=False)
            data=json.loads((mod.HERE/'crn_development_losses.json').read_text())
            assert data['split']=='smoke_only' and len(data['scenarios'])==2
            for v in data['scenarios'].values():
                assert len(v['path_ids'])==16
                assert len(v['losses'])==5
            print('PASS mocked source-backend smoke: arrays, scores, output schema; not real estimator')
        finally:
            mod.load_backend=old;mod.HERE=old_here

if __name__=='__main__':main()
