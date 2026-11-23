"""Source-compatible FRESH 50-node common-G development runner using Watson CRN.

Must be placed next to the frozen paper1_* folders in the user's paperlast
repository. Never reads or writes the previous SSCD v2 development/test files.
Without --execute only performs dry-run parameter checks. --smoke never locks.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXPECTED_HEAD_REFERENCE='32d27189a220ce28893bb68ac9b97cf85abeee8c'
SCENARIOS_ANGLES=(0.,-45.,45.)
SCENARIOS_START=(50,65)
CANDIDATES={'neg_boundary':-1.3152369768257723,'neg06':-.6,
            'iid':0.,'pos06':.6,'pos_boundary':1.3152369768257723}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_backend():
    src=ROOT/'paper1_common_g50_gain_acceptance_20261121'
    rep=ROOT/'paper1_50node_tv_gate_theory_20261121'
    dev=ROOT/'paper1_sscd_dev_protocol_20261122'
    for d in (src,rep,dev):
        if not d.is_dir():raise FileNotFoundError(f'Missing frozen research source {d}')
        sys.path.insert(0,str(d))
    import run_development as old
    from run_common_g_50 import load_common_g,physical_paths,run_common_g
    from calibration_rule import path_fwer_cutoff
    old.validate_fixtures()  # exact 50-node source hashes frozen by prior audit
    return old,load_common_g,physical_paths,run_common_g,path_fwer_cutoff


def unique_seeds(master,scenario,model_index):
    j=model_index*6+scenario
    return {'physical':master+1_000_000+j*1_000_000,
            'radial':master+20_000_000+j*1_000_000,
            'initial':master+40_000_000+j*1_000_000,
            'quantile_and_sign':master+60_000_000+j*1_000_000}


def evaluate_calibration(label,model,n,seed,physical_paths,run_common_g,cutoff_fn):
    phy=physical_paths(model,n,seed)
    out=run_common_g(model,phy,gate_on=False)
    vals=np.max(out.scores[:,4:],axis=(1,2))
    rule=cutoff_fn(vals,.05)
    return {'threshold':rule,'max_scores':vals.tolist(),
            'path_ids':[f'crn_cal_{seed}_{label}_{i}' for i in range(n)]}


def run(*,seed,cal_paths,dev_paths,smoke=False,allow_overwrite=False):
    from watson_crn import make_common_draws,beta_from_common,watermark_attack_from_beta,theoretical_zeta_mi
    from selector_v3 import select
    old,load,physical_paths,common_g,cutoff=load_backend()
    if cal_paths<100 or dev_paths<16:raise ValueError('cal_paths>=100, dev_paths>=16 required')
    output=HERE/'crn_development_losses.json'
    if output.exists() and not allow_overwrite:raise FileExistsError(f'{output}: refusing to overwrite existing research data')
    angles=SCENARIOS_ANGLES if not smoke else (0.,)
    onsets=SCENARIOS_START if not smoke else (50,)
    inputs={'split':'smoke_only' if smoke else 'development_crn','dimension':2,
      'theta':CANDIDATES,'scenarios':{},'source_reference':EXPECTED_HEAD_REFERENCE}
    metadata={'experiment':'firstpaper_CRN_frozen_commonG_selection','source_head_reference':EXPECTED_HEAD_REFERENCE,
      'frozen_configuration_sha256':old.validate_fixtures(),
      'crn_generator_sha256':sha(HERE/'watson_crn.py'),
      'run_crn_development_sha256':sha(__file__),
      'selector_v3_sha256':sha(HERE/'selector_v3.py'),
      'seed':seed,'calibration_count':cal_paths,'paths_per_scenario':dev_paths,
      'candidate_theta':CANDIDATES,
      'sampling':'signed-von-Mises quantile CRN: same radial/initial/quantile/branch across policy, independent across scenarios',
      'attack':'original six A0 edges only, same fixed joint raw energy per angle',
      'calibrations':{},'scenarios':{},
      'limitations':'no unknown support, new selector requires separate final testing',
      'smoke_only':smoke}
    seed_registry=set()
    for mi,label in enumerate(('unstable','stable')):
        model=load(label)
        cal_seed=seed+100_000_000+mi*10_000_000
        cal=evaluate_calibration(label,model,cal_paths,cal_seed,physical_paths,common_g,cutoff)
        threshold=cal['threshold']['threshold']
        metadata['calibrations'][label]=cal
        for ai,angle in enumerate(angles):
            attack,scale=old.rotated_raw_attack(model,angle)
            for oi,onset in enumerate(onsets):
                ix=ai*2+oi
                sd=unique_seeds(seed,ix,mi)
                if any(s in seed_registry for s in sd.values()):raise AssertionError('RNG seed collision')
                seed_registry.update(sd.values())
                phy=physical_paths(model,dev_paths,sd['physical'])
                d=make_common_draws(dev_paths,60,len(model.links),
                       sd['radial'],sd['initial'],sd['quantile_and_sign'])
                losses={};affected={}
                for name,th in CANDIDATES.items():
                    beta=beta_from_common(d,th)
                    delta=watermark_attack_from_beta(attack[50:110],beta,T=model.T,offset=onset)
                    w=common_g(model,phy,delta,gate_on=True,threshold=threshold)
                    squared=np.sum((w.post-phy.x[:,:,None,:])**2,axis=-1)
                    losses[name]=squared[:,20:].mean(axis=(1,2)).tolist()
                    affected[name]=squared[:,20:,model.receiver_ids].mean(axis=(1,2)).tolist()
                scen=f'{label}__rot{angle:+.0f}__onset{onset}'
                inputs['scenarios'][scen]={'path_ids':[f'crn_dev_{seed}_{scen}_{j}' for j in range(dev_paths)],
                    'losses':losses,'affected_losses':affected}
                metadata['scenarios'][scen]={'onset':onset,'angle':angle,'seed_namespaces':sd,
                    'attack_scale':scale,'raw_attack_joint_energy':float(np.sum(attack**2)),
                    'threshold':threshold}
                print('completed CRN development scenario:',scen,flush=True)
    output.write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    metadata['development_sha256']=sha(output)
    if not smoke:
        # These thresholds are new proposed design choices, not selections
        # retroactively inferred from the historic held-out test.
        choice=select(inputs,information_budget=.1,
           noninferiority_margin=.001,minimum_average_benefit=.001,alpha=.05)
        lock=HERE/'crn_locked_selection.json'
        if lock.exists() and not allow_overwrite:raise FileExistsError('Locked selection already exists')
        lock.write_text(json.dumps(choice,indent=2)+'\n',encoding='utf-8')
        metadata['lock_sha256']=sha(lock)
        metadata['selected']=choice['selection']
    else:metadata['selected']=None
    (HERE/'crn_development_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'CRN_DEVELOPMENT_COMPLETE' if not smoke else 'CRN_SMOKE_ONLY',
      'selected':metadata['selected'],'scenario_count':len(inputs['scenarios'])},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--seed',type=int,default=2026120302)
    p.add_argument('--calibration',type=int,default=512)
    p.add_argument('--paths',type=int,default=256)
    p.add_argument('--allow-overwrite',action='store_true')
    a=p.parse_args()
    if not a.execute:
        from watson_crn import theoretical_zeta_mi
        print(json.dumps({'status':'DRY_RUN_NO_50_NODE_SIM','new_seed':a.seed,
                          'planned_scenarios':12,'theta_mi':{k:theoretical_zeta_mi(v)[1] for k,v in CANDIDATES.items()},
                          'previous_locked_choice':'iid; never overwrite previous historical artifacts'},indent=2))
        return
    run(seed=a.seed,cal_paths=a.calibration,dev_paths=a.paths,smoke=a.smoke,allow_overwrite=a.allow_overwrite)

if __name__=='__main__':main()
