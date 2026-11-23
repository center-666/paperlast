"""Independent final test for frozen CRN-SSCD v3 selection (first paper only).

Only callable after a NEW independent development run has produced a lock.
Tests all fixed watermark families descriptively; NEVER reselects using test.
The historical 32d2718 iid lock is not read, changed or superseded.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
from watson_crn import make_common_draws,beta_from_common,watermark_attack_from_beta
from run_crn_development import load_backend,CANDIDATES,SCENARIOS_ANGLES,SCENARIOS_START,sha,unique_seeds
from selector_v3 import select


def measure(world,phy,ids,onset,threshold):
    err=np.sum((world.post-phy.x[:,:,None,:])**2,axis=-1)
    net=err.mean(axis=2)
    aff=err[:,:,ids].mean(axis=2)
    start,stop=onset,onset+60
    scores=world.scores>threshold
    decision=scores.any(axis=-1)
    whole=decision[:,start:stop]
    eligible=decision[:,start+7:stop]
    def delay(a,full_length):
        return np.where(a.any(axis=1),np.argmax(a,axis=1),full_length)
    return {'network_post20':net[:,20:].mean(axis=1),
      'affected_post20':aff[:,20:].mean(axis=1),
      'network_attack':net[:,start:stop].mean(axis=1),
      'affected_attack':aff[:,start:stop].mean(axis=1),
      'network_postattack':net[:,stop:].mean(axis=1),
      'affected_postattack':aff[:,stop:].mean(axis=1),
      'eligible_link_time_alarm_fraction':scores[:,start+7:stop,:].mean(axis=(1,2)),
      'attack_path_any_alarm':whole.any(axis=1).astype(float),
      'raw_first_alarm_delay_censored60':delay(whole,60),
      'eligible_first_alarm_delay_from_onset':np.where(eligible.any(axis=1),7+np.argmax(eligible,axis=1),60),
      'drop_fraction_attack':world.gates[:,start:stop].mean(axis=(1,2))}


def smean(a):
    x=np.asarray(a,float)
    return {'mean':float(x.mean()),'path_se':float(x.std(ddof=1)/np.sqrt(len(x))),
            'n_paths':len(x)}


def run(*,seed=2026120705,cal_paths=512,paths=512,allow_overwrite=False):
    base=HERE/'crn_development_losses.json'
    lock_path=HERE/'crn_locked_selection.json'
    manifest_path=HERE/'crn_development_manifest.json'
    for f in (base,lock_path,manifest_path):
        if not f.is_file():raise FileNotFoundError(f'Fresh development must be completed before final evaluation: {f}')
    dev=json.loads(base.read_text());lock=json.loads(lock_path.read_text());man=json.loads(manifest_path.read_text())
    if dev.get('split')!='development_crn' or lock.get('status')!='FRESH_DEVELOPMENT_DESIGN_ONLY_NOT_A_TEST':
        raise ValueError('Expected a fresh v3 development lock (not historical 32d2718)')
    if sha(base)!=man['development_sha256'] or sha(lock_path)!=man['lock_sha256']:
        raise ValueError('Development/lock hashes no longer match')
    if json.dumps(dev['theta'],sort_keys=True)!=json.dumps(CANDIDATES,sort_keys=True):
        raise ValueError('Candidate family changed after lock')
    if sha(HERE/'watson_crn.py')!=man['crn_generator_sha256']:
        raise ValueError('CRN generator changed after development')
    if sha(HERE/'selector_v3.py')!=man['selector_v3_sha256']:
        raise ValueError('Selection algorithm changed after development')
    if lock['selection'] not in CANDIDATES or lock['theta_locked']!=CANDIDATES[lock['selection']]:
        raise ValueError('Lock is inconsistent with candidate family')
    if seed==man['seed'] or paths<16 or cal_paths<100:
        raise ValueError('Test seeds and path counts invalid')
    result_path=HERE/'crn_final_results.json'
    archive_path=HERE/'crn_final_path_arrays.npz'
    if (result_path.exists() or archive_path.exists()) and not allow_overwrite:
        raise FileExistsError('Refusing to overwrite existing final-test JSON or path archive')
    old,load,phy_paths,common_g,cutoff=load_backend()
    old.validate_fixtures()
    result={'status':'INDEPENDENT_TEST_EXECUTED_NO_RESELECTION',
            'historical_lock_32d2718':'iid (untouched)',
            'new_v3_selection':lock['selection'],'new_v3_theta':lock['theta_locked'],
            'source_reference':man['source_head_reference'],
            'test_seed':seed,'calibration_paths_per_model':cal_paths,
            'test_paths_per_scenario':paths,'models':{},
            'warning':'Other fixed candidates are descriptive; never use them to change this lock'}
    arrays={}
    for mi,label in enumerate(('unstable','stable')):
        model=load(label)
        cal_seed=seed+500_000_000+mi*10_000_000
        ph_cal=phy_paths(model,cal_paths,cal_seed)
        w_cal=common_g(model,ph_cal,gate_on=False)
        calmax=np.max(w_cal.scores[:,4:],axis=(1,2))
        th_record=cutoff(calmax,.05)
        threshold=float(th_record['threshold'])
        ht_seed=seed+600_000_000+mi*10_000_000
        healthy=phy_paths(model,paths,ht_seed)
        hw=common_g(model,healthy,gate_on=True,threshold=threshold)
        hit=hw.gates[:,4:].any(axis=(1,2))
        k=int(hit.sum());N=len(hit)
        ci=[float(beta_dist.ppf(.025,k,N-k+1)) if k else 0.,
            float(beta_dist.ppf(.975,k+1,N-k)) if k<N else 1.]
        mh={'calibration_threshold':threshold,'calibration_rank':th_record,
            'heldout_healthy_FWER':float(hit.mean()),'healthy_alarms':k,
            'healthy_FWER_ClopperPearson_CI95':ci,
            'calibration_seed':cal_seed,
            'calibration_path_ids':[f'crn_final_cal_{cal_seed}_{label}_{i}' for i in range(cal_paths)],
            'healthy_test_seed':ht_seed,
            'healthy_test_path_ids':[f'crn_final_healthy_{ht_seed}_{label}_{i}' for i in range(paths)],
            'healthy_network_mse_post20':smean(np.sum((hw.post-healthy.x[:,:,None,:])**2,axis=-1)[:,20:].mean(axis=(1,2))),
            'scenarios':{}}
        arrays[f'{label}_calibration_maxscore']=calmax
        arrays[f'{label}_healthy_test_any_alarm']=hit.astype(np.uint8)
        for ai,angle in enumerate(SCENARIOS_ANGLES):
            raw_attack,scale=old.rotated_raw_attack(model,angle)
            for oi,onset in enumerate(SCENARIOS_START):
                scen=f'{label}__rot{angle:+.0f}__onset{onset}'
                sd=unique_seeds(seed+700_000_000,ai*2+oi,mi)
                physical=phy_paths(model,paths,sd['physical'])
                d=make_common_draws(paths,60,len(model.links),sd['radial'],sd['initial'],sd['quantile_and_sign'])
                score={};rec={}
                for p,theta in CANDIDATES.items():
                    beta=beta_from_common(d,theta)
                    delta=watermark_attack_from_beta(raw_attack[50:110],beta,T=model.T,offset=onset)
                    w=common_g(model,physical,delta,gate_on=True,threshold=threshold)
                    vals=measure(w,physical,model.receiver_ids,onset,threshold)
                    rec[p]=vals;score[p]={metric:smean(x) for metric,x in vals.items()}
                    for metric,x in vals.items(): arrays[f'{scen}_{p}__{metric}']=x
                gate=np.zeros((paths,model.T,len(model.links)),bool);gate[:,1:]=True
                oracle=common_g(model,physical,np.zeros((paths,model.T,len(model.links),2)),
                                  gate_override=gate,threshold=threshold)
                oracle_m=measure(oracle,physical,model.receiver_ids,onset,threshold)
                for metric,x in oracle_m.items():arrays[f'{scen}_oracle_static_six__{metric}']=x
                mh['scenarios'][scen]={'attack_angle':angle,'attack_onset':onset,
                    'attack_raw_energy':float(np.sum(raw_attack**2)),
                    'calibration_threshold':threshold,'seed_namespaces':sd,
                    'test_path_ids':[f'crn_final_{seed}_{scen}_{i}' for i in range(paths)],
                    'attack_window_half_open':[onset,onset+60],
                    'eligible_score_window_half_open':[onset+7,onset+60],
                    'post_attack_window_half_open':[onset+60,150],
                    'candidate_means':score,
                    'oracle_static_six':{k:smean(x) for k,x in oracle_m.items()},
                    'paired_vs_iid':{p:{key:smean(vals-rec['iid'][key]) for key,vals in pvalues.items()}
                                     for p,pvalues in rec.items() if p!='iid'}}
                print('completed CRN final scenario',scen,flush=True)
        result['models'][label]=mh
    np.savez_compressed(archive_path,**arrays)
    result['lock_sha256']=sha(lock_path);result['path_archive_sha256']=sha(archive_path)
    result['run_script_sha256']=sha(__file__)
    result_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'selected_unchanged':result['new_v3_selection'],
                      'scenario_count':12,'paths_per_scenario':paths},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--seed',type=int,default=2026120705)
    p.add_argument('--calibration',type=int,default=512)
    p.add_argument('--paths',type=int,default=512)
    p.add_argument('--allow-overwrite',action='store_true')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({'status':'DRY_RUN_NO_TEST','needs':'a fresh crn_locked_selection.json generated under a new development seed',
                          'fixed_scenarios':12,'future_test_seed':a.seed},indent=2))
        return
    run(seed=a.seed,cal_paths=a.calibration,paths=a.paths,allow_overwrite=a.allow_overwrite)

if __name__=='__main__':main()
