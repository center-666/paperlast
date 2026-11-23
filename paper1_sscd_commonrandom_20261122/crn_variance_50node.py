"""Paired-SE audit of common transition draws on the recovered 50-node system.

CRN is used only to couple offline estimator experiments. The compared
watermark laws retain their individual marginal distributions. This audit is
not part of parameter selection, and its data are kept separate from the v3
development and final-test splits.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'paper1_common_g50_gain_acceptance_20261121'))
sys.path.insert(0,str(ROOT/'paper1_50node_tv_gate_theory_20261121'))
from watson_crn import make_common_draws,beta_from_common,watermark_attack_from_beta
from run_crn_development import load_backend

REPS=8
PATHS=512
CALIBRATION=512
THETA_NEG=-1.3152369768257723
MASTER_SEED=2026120805


def mse_paths(world,physical):
    error=np.sum((world.post-physical.x[:,:,None,:])**2,axis=-1)
    return error[:,20:].mean(axis=(1,2))


def paired_stats(values):
    x=np.asarray(values,float)
    return {'mean':float(x.mean()),
            'path_se':float(x.std(ddof=1)/np.sqrt(len(x))),
            'paths':int(len(x))}


def main():
    for name in ('crn_variance_50node_results.json','crn_variance_50node_paths.npz'):
        if (HERE/name).exists():
            raise FileExistsError(f'Refusing to overwrite existing CRN audit: {HERE/name}')
    old,load,physical_paths,common_g,cutoff_fn=load_backend()
    source=ROOT/'paper1_common_g50_gain_acceptance_20261121'
    result={'status':'PASS','kind':'REAL_50_NODE_CRN_PAIRED_SE_AUDIT',
            'scope':'Original six-edge angle-zero attack; separate from v3 development/final test.',
            'paths_per_replication':PATHS,'replications':REPS,
            'healthy_calibration_paths_per_model':CALIBRATION,
            'seed':MASTER_SEED,'policies':{'iid':0.0,'negative':THETA_NEG},
            'sharing_intervention':'Same physical path, watermark radial variables and initial directions; CRN pair shares transition quantiles/branch bits; independent pair uses distinct transition streams.',
            'models':{},
            'source_hashes':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                             for name in ['gain_table.json','physical_configuration.json','run_common_g_50.py']}}
    archive={}
    for model_i,label in enumerate(['unstable','stable']):
        model=load(label)
        cal=physical_paths(model,CALIBRATION,MASTER_SEED+1_000_000+model_i*100_000)
        cal_scores=common_g(model,cal,gate_on=False).scores
        calibration=cutoff_fn(np.max(cal_scores[:,4:],axis=(1,2)),.05)
        threshold=float(calibration['threshold'])
        records=[]
        for rep in range(REPS):
            seed=MASTER_SEED+10_000_000+model_i*1_000_000+rep*10_000
            physical=physical_paths(model,PATHS,seed+1)
            radial_seed,initial_seed=seed+2,seed+3
            crn_draws=make_common_draws(PATHS,60,len(model.links),radial_seed,initial_seed,seed+4)
            iid_ind_draws=make_common_draws(PATHS,60,len(model.links),radial_seed,initial_seed,seed+5)
            neg_ind_draws=make_common_draws(PATHS,60,len(model.links),radial_seed,initial_seed,seed+6)
            attack=model.attack[50:110]
            deltas={
                'crn_iid':watermark_attack_from_beta(attack,beta_from_common(crn_draws,0.),model.T,50),
                'crn_negative':watermark_attack_from_beta(attack,beta_from_common(crn_draws,THETA_NEG),model.T,50),
                'independent_iid':watermark_attack_from_beta(attack,beta_from_common(iid_ind_draws,0.),model.T,50),
                'independent_negative':watermark_attack_from_beta(attack,beta_from_common(neg_ind_draws,THETA_NEG),model.T,50),
            }
            losses={}
            for name,delta in deltas.items():
                world=common_g(model,physical,delta,gate_on=True,threshold=threshold)
                losses[name]=mse_paths(world,physical)
                archive[f'{label}_{name}__mse_rep{rep}']=losses[name]
            d_crn=losses['crn_negative']-losses['crn_iid']
            d_ind=losses['independent_negative']-losses['independent_iid']
            se_crn=float(d_crn.std(ddof=1)/np.sqrt(PATHS))
            se_ind=float(d_ind.std(ddof=1)/np.sqrt(PATHS))
            records.append({'replication':rep,'seed_namespace_base':seed,
                            'mean_difference_crn':float(d_crn.mean()),
                            'mean_difference_independent':float(d_ind.mean()),
                            'paired_path_se_crn':se_crn,
                            'paired_path_se_independent':se_ind,
                            'se_ratio_crn_over_independent':se_crn/se_ind,
                            'effective_sample_ratio_independent_over_crn':(se_ind/se_crn)**2})
            archive[f'{label}_crn_difference_rep{rep}']=d_crn
            archive[f'{label}_independent_difference_rep{rep}']=d_ind
            print(label,'rep',rep+1,'CRN SE',round(se_crn,8),'independent SE',round(se_ind,8),flush=True)
        crn_ses=np.array([x['paired_path_se_crn'] for x in records])
        ind_ses=np.array([x['paired_path_se_independent'] for x in records])
        ratio=crn_ses/ind_ses
        mean_difference=np.array([x['mean_difference_crn']-x['mean_difference_independent'] for x in records])
        result['models'][label]={
            'threshold':calibration,
            'mean_crn_paired_se':paired_stats(crn_ses),
            'mean_independent_paired_se':paired_stats(ind_ses),
            'mean_paired_se_ratio':float(ratio.mean()),
            'mean_relative_se_reduction_percent':float(100*(1-ratio.mean())),
            'mean_effective_sample_ratio_independent_over_crn':float(np.mean((ind_ses/crn_ses)**2)),
            'replication_mean_difference_crn_minus_independent':paired_stats(mean_difference),
            'replications':records,
        }
    np.savez_compressed(HERE/'crn_variance_50node_paths.npz',**archive)
    result['path_archive_sha256']=hashlib.sha256((HERE/'crn_variance_50node_paths.npz').read_bytes()).hexdigest()
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'crn_variance_50node_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({label:{k:v for k,v in rec.items() if k in ['mean_relative_se_reduction_percent','mean_effective_sample_ratio_independent_over_crn','mean_crn_paired_se','mean_independent_paired_se']} for label,rec in result['models'].items()},indent=2))


if __name__=='__main__':main()
