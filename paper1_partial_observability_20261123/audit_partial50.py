"""Read back partial-observation results and independently verify all gains.

Also runs a clearly post-hoc stronger baseline that knows the attack interval.
It never selects or modifies watermark parameters or the original results.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import t as student_t
from run_partial50 import (ROOT,HERE,sha,design,physical,run_filter,measure,summary,obs_rank)


def main():
    result=json.loads((HERE/'partial50_results.json').read_text(encoding='utf-8'))
    reg=json.loads((HERE/'registration.json').read_text(encoding='utf-8'))
    assert result['registration_sha256']==sha(HERE/'registration.json')
    assert result['script_sha256']==sha(HERE/'run_partial50.py')
    assert result['path_archive_sha256']==sha(HERE/'partial50_paths.npz')
    cfg=json.loads((ROOT/reg['frozen_input'].replace('../','',1)).read_text(encoding='utf-8'))
    values=np.load(HERE/'partial50_paths.npz',allow_pickle=False)
    report={'status':'PASS','evidence':'Exploratory n=4,m=2 physical extension, original six directed attack supports',
            'readback_path_arrays':len(values.files),'original_results_unmodified':True,'models':{},
            'intervals':'Conditional on the realized calibration; unadjusted approximate t intervals. Additional 6-comparison network-MSE Bonferroni intervals are descriptive.'}
    derived={'source_configuration_sha256':reg['input_sha256'],'graph':cfg['graph'],
             'attack_links':cfg['attack_links'],'message_dimension':2,'state_dimension':4,
             'R':cfg['R'],'models':{}}
    extra={};extra_path=HERE/'timing_oracle_paths.npz';extra_result=HERE/'timing_oracle_results.json'
    cached=json.loads(extra_result.read_text(encoding='utf-8')) if extra_result.exists() else None
    if cached is not None:
        assert extra_path.exists() and cached['path_archive_sha256']==sha(extra_path)
    else:
        if extra_path.exists():raise FileExistsError('Unpaired timing-oracle archive; refusing overwrite')
    largest_gain_error=0.;largest_F_error=0.
    for label in ('unstable','stable'):
        record=result['models'][label];model,recomputed=design(label,cfg)
        saved=json.loads((HERE/f'{label}_gain_design.json').read_text(encoding='utf-8'))
        assert saved==recomputed
        derived['models'][label]={'A':model.A.tolist(),'Q':model.Q.tolist(),'H':model.H.tolist(),
                                   'node_subsystem_ids':model.types.tolist()}
        for row in saved['rows']:
            i=row['receiver'];active=row['active_neighbors']
            C=np.vstack([model.H[i],model.H[active].sum(axis=0)])
            V=block_diag(model.R[i],model.R[active].sum(axis=0))
            assert obs_rank(model.A,C)==4
            P=np.eye(4)
            for iteration in range(5000):
                prior=model.A@P@model.A.T+model.Q
                L=np.linalg.solve(C@prior@C.T+V,C@prior).T
                next_P=prior-L@C@prior
                if np.linalg.norm(next_P-P,'fro')<1e-12:break
                P=next_P
            else:raise ArithmeticError('Independent posterior Riccati iteration did not converge')
            Lsaved=np.hstack([row['K'],row['G']])
            largest_gain_error=max(largest_gain_error,float(np.max(np.abs(L-Lsaved))))
            FF=(np.eye(4)-Lsaved@C)@model.A
            largest_F_error=max(largest_F_error,float(np.max(np.abs(FF-row['F']))))
        cal=values[f'{label}__calibration_maxima'];n=len(cal)
        assert n==reg['calibration_paths_per_model']==512
        assert np.partition(cal,487)[487]==record['threshold']
        hits=values[f'{label}__healthy_any_alarm'];assert int(hits.sum())==record['healthy_alarms']
        for policy,metrics in record['attacked'].items():
            for metric,stats in metrics.items():
                x=values[f'{label}__{policy}__{metric}']
                assert x.shape==(512,) and np.isfinite(x).all()
                assert abs(x.mean()-stats['mean'])<1e-12
        iid=values[f'{label}__iid__network_post20'];neg=values[f'{label}__negative__network_post20']
        static=values[f'{label}__static_six_drop__network_post20']
        neg_static=summary(neg-static)
        se=np.std(neg-static,ddof=1)/np.sqrt(len(neg));critical=student_t.ppf(1-.05/(2*6),len(neg)-1)
        neg_static['descriptive_6_comparison_bonferroni_ci95']=[float((neg-static).mean()-critical*se),float((neg-static).mean()+critical*se)]
        healthy_all=record['healthy']['all_registered_retained']['network_post20']['mean']
        healthy_static=record['healthy']['static_six_drop']['network_post20']['mean']
        if cached is None:
            phy=physical(model,reg['attacked_paths_per_model'],record['seeds']['attack_physical'])
            schedule=np.zeros((len(phy.x),150,6),bool);schedule[:,50:110]=True
            world=run_filter(model,phy,np.zeros((len(phy.x),150,6,2)),record['threshold'],schedule)
            timing_metrics=measure(model,phy,world)
            for metric,v in timing_metrics.items():extra[f'{label}__timing_oracle__{metric}']=v
        else:
            with np.load(extra_path,allow_pickle=False) as saved_extra:
                timing_metrics={metric:saved_extra[f'{label}__timing_oracle__{metric}'].copy()
                                for metric in record['attacked']['iid']}
        report['models'][label]={'iid_network_mse':float(iid.mean()),'negative_network_mse':float(neg.mean()),
            'negative_relative_to_iid_percent':float(100*(neg.mean()/iid.mean()-1)),
            'negative_minus_iid':summary(neg-iid),'negative_minus_permanent_six_drop':neg_static,
            'negative_minus_known_support_and_timing_oracle':summary(neg-timing_metrics['network_post20']),
            'timing_oracle':{metric:summary(v) for metric,v in timing_metrics.items()},
            'permanent_drop_healthy_cost_percent':float(100*(healthy_static/healthy_all-1)),
            'healthy_fwer':record['healthy_path_fwer'],
            'attack_period_negative_minus_permanent':summary(values[f'{label}__negative__network_attack']-values[f'{label}__static_six_drop__network_attack']),
            'missing_mode_negative_minus_iid':summary(values[f'{label}__negative__missing_block_post20']-values[f'{label}__iid__missing_block_post20']),
            'retained_complementary_neighbor_minimum':saved['minimum_complementary_neighbors_in_registered_modes'],
            'max_norm_F':saved['max_norm_F_2']}
    assert largest_gain_error<1e-10 and largest_F_error<1e-13
    report['independent_gain_max_abs_difference']=largest_gain_error
    report['independent_F_reconstruction_max_abs_difference']=largest_F_error
    report['sources']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
        ROOT/'paper1_50node_tv_gate_theory_20261121/source_model.py',
        ROOT/'paper1_50node_tv_gate_theory_20261121/replay.py',
        ROOT/'paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py',
        ROOT/'paper1_sscd_commonrandom_20261122/watson_crn.py']}
    if cached is None:
        np.savez_compressed(extra_path,**extra)
        extra_result.write_text(json.dumps({'status':'POSTHOC_STRONGER_INFORMATION_BASELINE',
            'description':'Delete exactly the six known edges during the known attack interval [50,110), retain all outside. Uses extra attack-timing information, not a deployable detector and not used for selection.',
            'main_result_sha256':sha(HERE/'partial50_results.json'),'script_sha256':sha(__file__),
            'path_archive_sha256':sha(extra_path)},indent=2)+'\n',encoding='utf-8',newline='\n')
    (HERE/'derived_configuration.json').write_text(json.dumps(derived,indent=2)+'\n',encoding='utf-8',newline='\n')
    (HERE/'partial50_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
