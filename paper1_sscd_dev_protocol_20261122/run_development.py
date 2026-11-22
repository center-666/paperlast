"""First-paper common-G SSCD development-only experiment driver.

Source-compatible with frozen center-666/paperlast commit b9fc33c, but is NOT
an unseen-attack-support experiment and does NOT modify the estimator or detector.
Runs only when --execute is supplied. Do not feed earlier test paths to selector.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

THIS=Path(__file__).resolve().parent
ROOT=THIS.parent
sys.path.insert(0,str(THIS))
from selector_v2 import (REFERENCE_COMMIT, project_theta, select_from_development)

FIXTURE_HASHES={
  'gain_table.json':'57108c5f23138d5621138e4060e118144d34088b70bce1d9d3fc4a2914cd224e',
  'physical_configuration.json':'c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d',
  'run_common_g_50.py':'22ea30c33321da57aac0bf56b987221b529fc5020ada74b6cc789615f42d096c',
  'replay.py':'05e68f702ccea35d4bd5d4c1396c4acd9dbc79f4c6659150b1e283f279ddbabb'
}
REQUESTED={
  'neg_boundary':-1.315236976827179,
  'neg06':-.6,
  'iid':0.,
  'pos06':.6,
  'pos_boundary':1.315236976827179,
}
ATTACK_DURATION=60
PARITY_ORDER=4
SCORE_WINDOW=4
SCENARIO_ANGLES=(0.,-45.,45.)
SCENARIO_STARTS=(50,65)


def protocol_windows(onset,duration=ATTACK_DURATION,parity_order=PARITY_ORDER,score_window=SCORE_WINDOW):
    """Half-open attack, fully attack-eligible score, and post-attack windows."""
    if onset<0 or duration<=0 or onset+duration>150:
        raise ValueError('attack interval must fit the 150-step horizon')
    stop=onset+duration
    eligible_start=onset+parity_order+score_window-1
    if eligible_start>=stop:
        raise ValueError('attack too short for a fully attack-eligible score window')
    return {'attack':(onset,stop),'eligible_score':(eligible_start,stop),
            'post_attack':(stop,150),'eligible_offset_from_onset':eligible_start-onset,
            'eligible_delay_censor':stop-eligible_start}


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_fixtures():
    checked={}
    for name,expected in FIXTURE_HASHES.items():
        folder='paper1_50node_tv_gate_theory_20261121' if name=='replay.py' else 'paper1_common_g50_gain_acceptance_20261121'
        path=ROOT/folder/name
        if not path.is_file():raise FileNotFoundError(f'missing: {path}')
        found=sha256(path)
        if found != expected:
            raise RuntimeError(f'hash mismatch for {name}: {found} != {expected}')
        checked[name]=found
    return checked


def rotated_raw_attack(model, angle_degrees):
    """Same six registered edges and A0 virtual-state attack, rotated direction.

    The original joint link-domain attack energy is preserved; different angles
    are new PREDECLARED stress scenarios, not replicas of the original attack.
    """
    orig=np.asarray(model.attack,float)
    if orig.shape!=(150,6,2):raise ValueError('expected original six-link attack')
    if angle_degrees==0:return orig.copy(), 1.0
    H=np.asarray(model.H)
    links=np.asarray(model.links,int)
    stacked=np.vstack([H[j] for i,j in links])
    raw0=orig[50].reshape(-1)
    xi,residual,rank,sv=np.linalg.lstsq(stacked,raw0,rcond=None)
    if rank!=2 or np.linalg.norm(stacked@xi-raw0)>1e-7*max(1,np.linalg.norm(raw0)):
        raise ValueError('original attack cannot be reconstructed as a shared virtual state')
    a=np.deg2rad(angle_degrees)
    R=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    xi=R@xi
    target=np.zeros_like(orig)
    for h in range(ATTACK_DURATION):
        Ah=np.linalg.matrix_power(model.A,h)
        target[50+h]=np.asarray([H[j]@Ah@xi for i,j in links])
    original_E=float(np.sum(orig*orig));candidate_E=float(np.sum(target*target))
    if candidate_E<=0:raise ValueError('rotated attack has zero coded energy')
    scalar=math.sqrt(original_E/candidate_E)
    target*=scalar
    if not np.isclose(np.sum(target**2),original_E,rtol=1e-12):
        raise AssertionError('attack energy normalization failure')
    return target,scalar


def decoded_for_scenario(model, count, theta, attack_raw, start,
                         radial_seed, initial_seed, direction_seed, watermark_generator):
    """Generate coded A0 injection using exact legacy watermark generator.

    Rather than editing legacy replay.py, make a shallow model clone, replace
    only its attack waveform in source window [50,110), then shift the decoded
    injection to the requested PREDECLARED onset.
    """
    if not 1<=start<=150-ATTACK_DURATION:raise ValueError('invalid attack onset')
    cloned=SimpleNamespace(**vars(model))
    cloned.attack=np.asarray(attack_raw,float)
    delta,_=watermark_generator(cloned,count,theta,radial_seed,initial_seed,direction_seed)
    if delta.shape !=(count,150,6,2):raise ValueError('unexpected attack generator output')
    if start==50:return delta
    shifted=np.zeros_like(delta)
    shifted[:,start:start+ATTACK_DURATION]=delta[:,50:110]
    return shifted


def path_losses(world,physical,affected):
    sq=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
    return {'network_post20':sq[:,20:].mean(axis=(1,2)),
            'affected_post20':sq[:,20:,affected].mean(axis=(1,2))}


def scenario_rng_seeds(seed, model_i, angle_i, onset_i, policy_i):
    """Namespace independent streams by scenario and policy; pair only intended inputs."""
    scenario_index=model_i*6+angle_i*2+onset_i
    return {
        'physical':seed+1_000_000+1_000*scenario_index,
        'radial':seed+10_000_000+10_000*scenario_index,
        'initial':seed+20_000_000+10_000*scenario_index,
        'direction':seed+30_000_000+100_000*scenario_index+10_000*policy_i,
    }


def run(*, ncal=512, ndev=256, seed=2026112701, budget=.1,
        angle_set=SCENARIO_ANGLES, onset_set=SCENARIO_STARTS, smoke_only=False):
    hashes=validate_fixtures()
    for name in ('paper1_common_g50_gain_acceptance_20261121','paper1_50node_tv_gate_theory_20261121'):
        sys.path.insert(0,str(ROOT/name))
    from run_common_g_50 import load_common_g,physical_paths,run_common_g
    from replay import attack_watermarks
    from calibration_rule import path_fwer_cutoff
    if ndev<25 or ncal<100:raise ValueError('insufficient paths for development')
    candidates={}; audit={}
    for name,requested in REQUESTED.items():
        check=project_theta(requested,budget,m=2)
        audit[name]=check
        if check['status']=='feasible':candidates[name]=check['theta_used']
    if 'iid' not in candidates:raise RuntimeError('iid rejected')
    manifest={'source_commit':REFERENCE_COMMIT,'split':'development','seed':seed,
      'calibration_paths':ncal,'development_paths':ndev,
      'watermark_variance':.5,'attack_links':'original six registered',
      'attack_duration':ATTACK_DURATION,
      'scenario_angles_deg':angle_set,'scenario_start_times':onset_set,
      'rng_stream_policy':'Distinct nonoverlapping physical, radial, initial-direction and transition-direction seed namespaces; radial/initial shared across candidate policies only within a scenario.',
      'candidate_requested':REQUESTED,'candidate_adjustment':audit,
      'fixture_sha256':hashes,
      'development_runner_sha256':sha256(__file__),
      'development_selector_sha256':sha256(THIS/'selector_v2.py'),
      'split_policy':'New scenario-specific physical streams; prior 99ee54f, afde075 and b9fc33c test paths are not read or reused.',
      'note':'No earlier screening/test path is reused. Angle/onset variants are stress scenarios, not unknown attack support.'}
    all_losses={'source_commit':REFERENCE_COMMIT,'split':('smoke' if smoke_only else 'development'),
                'theta':candidates,'scenarios':{}}
    for mi,label in enumerate(('unstable','stable')):
        model=load_common_g(label)
        physical_cal=physical_paths(model,ncal,seed+100000*mi)
        cal_world=run_common_g(model,physical_cal,gate_on=False)
        cutoff=path_fwer_cutoff(np.max(cal_world.scores[:,4:],axis=(1,2)),.05)
        manifest[f'{label}_threshold']=cutoff
        for ai,angle in enumerate(angle_set):
            raw_attack,scale=rotated_raw_attack(model,angle)
            for oi,onset in enumerate(onset_set):
                scen=f'{label}__rot{angle:+.0f}__onset{onset}'
                streams=scenario_rng_seeds(seed,mi,ai,oi,0)
                phys=physical_paths(model,ndev,streams['physical'])
                path_ids=[f'dev_{seed}_{scen}_{k}' for k in range(ndev)]
                losses={};affected_losses={}
                # Pre-register common radii and initial directions across all policies.
                for pi,(policy,theta) in enumerate(candidates.items()):
                    streams=scenario_rng_seeds(seed,mi,ai,oi,pi)
                    delta=decoded_for_scenario(model,ndev,theta,raw_attack,onset,
                            streams['radial'],streams['initial'],streams['direction'],attack_watermarks)
                    world=run_common_g(model,phys,delta,gate_on=True,threshold=cutoff['threshold'])
                    loss=path_losses(world,phys,np.asarray(model.receiver_ids,int))
                    losses[policy]=loss['network_post20'].tolist()
                    affected_losses[policy]=loss['affected_post20'].tolist()
                all_losses['scenarios'][scen]={'path_ids':path_ids,'losses':losses,
                                               'affected_losses':affected_losses}
                manifest[scen]={'attack_scale':scale,'raw_joint_coded_energy':float(np.sum(raw_attack**2)),
                                'calibration_threshold':cutoff['threshold']}
                print('Completed development-only scenario',scen,flush=True)
    (THIS/'development_losses.json').write_text(json.dumps(all_losses,indent=2)+'\n',encoding='utf8')
    manifest['development_sha256']=sha256(THIS/'development_losses.json')
    if smoke_only:
        manifest['selected']=None
        manifest['warning']='SMOKE: no parameter was selected or locked'
        status='SMOKE_ONLY_NO_LOCK'
    else:
        lock=select_from_development(all_losses,budget=budget)
        (THIS/'locked_selection.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
        manifest['lock_sha256']=sha256(THIS/'locked_selection.json')
        manifest['selected']=lock['selected']
        status='DEVELOPMENT_COMPLETE_NOT_FINAL_EVALUATION'
    (THIS/'development_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':status, 'selected':manifest['selected'],
                      'scenarios':len(all_losses['scenarios'])},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--calibration',type=int,default=512)
    ap.add_argument('--paths',type=int,default=256)
    ap.add_argument('--seed',type=int,default=2026112702)
    ap.add_argument('--budget',type=float,default=.1)
    ap.add_argument('--single-scenario-smoke',action='store_true',
      help='Execute only original-angle and original-onset development smoke (not parameter lock).')
    args=ap.parse_args()
    if not args.execute:
        c={n:project_theta(t,args.budget) for n,t in REQUESTED.items()}
        print(json.dumps({'status':'DRY_RUN_NO_PHYSICAL_SIMULATION',
          'source_commit':REFERENCE_COMMIT,'eligible_theta':{k:v.get('theta_used') for k,v in c.items() if v['status']=='feasible'},
          'candidate_audit':c,'planned_scenarios':12,'seed':args.seed},indent=2))
        return
    if args.single_scenario_smoke:
        run(ncal=args.calibration,ndev=args.paths,seed=args.seed,
            angle_set=(0.,),onset_set=(50,),smoke_only=True,budget=args.budget)
    else:run(ncal=args.calibration,ndev=args.paths,seed=args.seed,budget=args.budget)

if __name__=='__main__':main()
