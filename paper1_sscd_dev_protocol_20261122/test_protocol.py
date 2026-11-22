"""Synthetic unit tests only; do not label them real 50-node results."""
import json
from types import SimpleNamespace
import numpy as np
from selector_v2 import (REFERENCE_COMMIT, information, project_theta,
                         select_from_development)
from run_development import rotated_raw_attack,decoded_for_scenario,scenario_rng_seeds,protocol_windows


def test_mi_projection():
    theta=-1.315236976827179
    before=information(theta)
    assert before>np.float64(.1)
    result=project_theta(theta,.1)
    assert result['status']=='feasible' and result['adjusted']
    assert information(result['theta_used'])<np.float64(.1)
    assert project_theta(2,.1)['status']=='over_budget'


def test_rotation_and_shift():
    # A simple source-compatible fake with 6 H_j blocks and the 150-step A0.
    A=np.array([[1.05,.02],[.01,1.03]])
    H=np.repeat(np.eye(2)[None],50,axis=0)
    links=np.array([[i,i+10] for i in range(6)],dtype=int)
    attack=np.zeros((150,6,2));xi=np.array([2.,-.4])
    for k in range(60):
        attack[50+k]=[H[j]@np.linalg.matrix_power(A,k)@xi for i,j in links]
    model=SimpleNamespace(A=A,H=H,links=links,attack=attack)
    untouched,scale=rotated_raw_attack(model,0)
    assert scale==1 and np.array_equal(untouched,attack)
    changed,scale=rotated_raw_attack(model,45)
    assert np.isclose(np.sum(changed**2),np.sum(attack**2))
    assert not np.allclose(changed,attack)

    def fake_wm(model,count,theta,rs,us,ds):
        d=np.broadcast_to(model.attack,(count,150,6,2)).copy()
        return d,np.zeros((count,60,6,2))
    d=decoded_for_scenario(model,3,-.6,changed,65,1,2,3,fake_wm)
    assert d.shape==(3,150,6,2)
    assert np.allclose(d[:,65:125],changed[50:110])
    assert np.all(d[:,:65]==0) and np.all(d[:,125:]==0)


def test_select_guards():
    rng=np.random.default_rng(1)
    p=[f'dev_{k}' for k in range(250)]
    a=rng.uniform(.05,.07,250)
    rec={'path_ids':p,'losses':{
        'iid':a.tolist(),
        'neg':(a-.001+rng.normal(size=250)*.0001).tolist(),
        'pos':(a+.001+rng.normal(size=250)*.0001).tolist()}}
    src={'source_commit':REFERENCE_COMMIT,'split':'development',
         'theta':{'iid':0.,'neg':-1.315236976827179,'pos':1.315236976827179},
         'scenarios':{'unstable':rec}}
    out=select_from_development(src)
    assert out['selected']=='neg'
    assert out['candidate_audit']['neg']['adjusted']
    assert all(s['simultaneous_baseline_lower']>0
               for s in out['candidate_audit']['neg']['per_scenario'])
    assert all(s['simultaneous_relative_upper_t_approx']<0
               for s in out['candidate_audit']['neg']['per_scenario'])
    bad={**src,'split':'test'}
    try:select_from_development(bad)
    except ValueError:pass
    else:raise AssertionError('Test split was accepted')
    rec2={**rec,'path_ids':p[:-1]+[p[0]]}
    bad={**src,'scenarios':{'bad':rec2}}
    try:select_from_development(bad)
    except ValueError:pass
    else:raise AssertionError('Duplicate path ID was accepted')
    rec3={'path_ids':p,'losses':{k:list(v) for k,v in rec['losses'].items()}}
    bad={**src,'scenarios':{'first':rec,'second':rec3}}
    try:select_from_development(bad)
    except ValueError:pass
    else:raise AssertionError('Cross-scenario path reuse was accepted')


def test_seed_namespaces():
    rows=[];calibration=[]
    for model_i in range(2):
        calibration.append(2026112702+100_000*model_i)
        for angle_i in range(3):
            for onset_i in range(2):
                for policy_i in range(5):
                    rows.append((model_i,angle_i,onset_i,policy_i,
                                 scenario_rng_seeds(2026112702,model_i,angle_i,onset_i,policy_i)))
    for kind in ['physical','radial','initial']:
        by_scenario={}
        for model_i,angle_i,onset_i,policy_i,streams in rows:
            by_scenario.setdefault((model_i,angle_i,onset_i),set()).add(streams[kind])
        assert len(by_scenario)==12 and all(len(v)==1 for v in by_scenario.values()),kind
        assert len({next(iter(v)) for v in by_scenario.values()})==12,kind
    directions=[entry[4]['direction'] for entry in rows]
    assert len(directions)==60 and len(set(directions))==60
    # Radial/initial seeds intentionally repeat five times within one paired scenario.
    unique_across_namespaces=[]
    for kind in ['physical','radial','initial','direction']:
        unique_across_namespaces.extend(set(entry[4][kind] for entry in rows))
    unique_across_namespaces.extend(calibration)
    assert len(unique_across_namespaces)==len(set(unique_across_namespaces))
    assert max(unique_across_namespaces)<2**32


def test_onset_specific_windows():
    expected={50:((50,110),(57,110),(110,150)),
              65:((65,125),(72,125),(125,150))}
    for onset,want in expected.items():
        got=protocol_windows(onset)
        assert (got['attack'],got['eligible_score'],got['post_attack'])==want
        scores=np.zeros((1,150,1))
        scores[0,onset+6,0]=1.0   # before all attack-affected scores
        scores[0,onset+7,0]=1.0   # first fully eligible score
        scores[0,onset+59,0]=1.0  # final in-support score
        scores[0,onset+60,0]=1.0 # after support
        alarms=scores[0,:,0]>0
        start,stop=got['eligible_score']
        assert alarms[start:stop].sum()==2
        assert alarms[stop]  # Present in raw score, excluded by half-open window.

if __name__=='__main__':
    test_mi_projection();test_rotation_and_shift();test_select_guards();test_seed_namespaces();test_onset_specific_windows()
    print(json.dumps({'status':'PASS','tests':5,
                      'note':'SYNTHETIC ONLY, no physical 50-node evaluation'},indent=2))
