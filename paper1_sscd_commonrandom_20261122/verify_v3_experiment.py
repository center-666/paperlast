"""Verify the completed v3 development, lock, final test and CRN SE audit."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import numpy as np
from selector_v3 import select as select_v3

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dev=json.loads((HERE/'crn_development_losses.json').read_text(encoding='utf-8'))
    dm=json.loads((HERE/'crn_development_manifest.json').read_text(encoding='utf-8'))
    lock=json.loads((HERE/'crn_locked_selection.json').read_text(encoding='utf-8'))
    final=json.loads((HERE/'crn_final_results.json').read_text(encoding='utf-8'))
    final_analysis=json.loads((HERE/'v3_final_analysis.json').read_text(encoding='utf-8'))
    path_manifest=json.loads((HERE/'crn_final_path_manifest.json').read_text(encoding='utf-8'))
    dev_arrays={}
    assert dev['split']=='development_crn' and dm['seed']==2026120302
    assert sha(HERE/'crn_development_losses.json')==dm['development_sha256']
    assert sha(HERE/'crn_locked_selection.json')==dm['lock_sha256']==final['lock_sha256']
    assert sha(HERE/'crn_development_losses.json')==dm['development_sha256']
    assert sha(HERE/'run_crn_development.py')==dm['run_crn_development_sha256']
    assert sha(HERE/'selector_v3.py')==dm['selector_v3_sha256']
    assert sha(HERE/'watson_crn.py')==dm['crn_generator_sha256']
    assert lock['selection']==final['new_v3_selection']=='neg_boundary'
    assert lock['theta_locked']==final['new_v3_theta']
    assert len(dev['scenarios'])==12
    recomputed_lock=select_v3(dev,information_budget=.1,noninferiority_margin=.001,
                              minimum_average_benefit=.001,alpha=.05)
    assert recomputed_lock['selection']==lock['selection']
    assert recomputed_lock['theta_locked']==lock['theta_locked']
    assert recomputed_lock['data_sha256']==lock['data_sha256']
    for model,cal in dm['calibrations'].items():
        maxima=np.asarray(cal['max_scores'])
        assert maxima.shape==(512,)
        assert np.isclose(np.partition(maxima,487)[487],cal['threshold']['threshold'],rtol=0,atol=1e-12)
    dev_ids=[]
    for name,sc in dev['scenarios'].items():
        ids=sc['path_ids'];assert len(ids)==len(set(ids))==256
        dev_ids.extend(ids)
        assert set(sc['losses'])==set(dm['candidate_theta'])
        for values in sc['losses'].values():assert len(values)==256 and np.isfinite(values).all()
    assert len(dev_ids)==len(set(dev_ids))==12*256
    assert len(final['models'])==2 and len(path_manifest['models'])==2
    arrays=np.load(HERE/'crn_final_path_arrays.npz',allow_pickle=False)
    assert sha(HERE/'crn_final_path_arrays.npz')==final['path_archive_sha256']
    with np.load(HERE/'crn_variance_50node_paths.npz',allow_pickle=False) as variance_paths:
        variance_path_count=len(variance_paths.files)
    with ZipFile(HERE/'source_package.zip') as z:
        zip_members={name:hashlib.sha256(z.read('paper1_sscd_commonrandom_20261122/'+name)).hexdigest()
                     for name in ['run_crn_development.py','run_crn_final.py','selector_v3.py','watson_crn.py']}
        zip_sources={name:z.read('paper1_sscd_commonrandom_20261122/'+name).decode('utf-8')
                     for name in ['run_crn_development.py','selector_v3.py','watson_crn.py']}
        original_zip_hash=hashlib.sha256((HERE/'source_package.zip').read_bytes()).hexdigest()
        archived_runner=zip_members['run_crn_final.py']
    assert archived_runner==final['run_script_sha256']
    assert archived_runner==sha(HERE/'execution_snapshot/run_crn_final_executed.py')
    for member,key in [('run_crn_development.py','run_crn_development_sha256'),
                       ('selector_v3.py','selector_v3_sha256'),('watson_crn.py','crn_generator_sha256')]:
        assert zip_members[member]==dm[key]
        archived_text=zip_sources[member]
        workspace_text=(HERE/member).read_text(encoding='utf-8')
        assert workspace_text==archived_text.replace('\r\n','\n').replace('\r','\n')
    assert path_manifest['status']=='PASS'
    for model in ['unstable','stable']:
        record=final['models'][model];manifest=path_manifest['models'][model]
        maxscores=arrays[f'{model}_calibration_maxscore']
        assert maxscores.shape==(512,)
        threshold=record['calibration_threshold']
        assert np.isclose(np.partition(maxscores,487)[487],threshold,rtol=0,atol=1e-12)
        assert len(manifest['calibration_path_ids'])==512
        assert len(manifest['healthy_test_path_ids'])==512
        all_ids=list(manifest['calibration_path_ids'])+list(manifest['healthy_test_path_ids'])
        assert len(set(all_ids))==len(all_ids)
        assert int(arrays[f'{model}_healthy_test_any_alarm'].sum())==record['healthy_alarms']
        assert set(record['scenarios'])==set(manifest['scenarios'])
        for scenario,sdata in record['scenarios'].items():
            ids=manifest['scenarios'][scenario]['path_ids']
            assert len(ids)==len(set(ids))==512 and not(set(ids)&set(all_ids))
            all_ids.extend(ids)
            for policy,means in sdata['candidate_means'].items():
                for metric,summary in means.items():
                    value=arrays[f'{scenario}_{policy}__{metric}']
                    assert value.shape==(512,) and np.isfinite(value).all()
                    assert np.isclose(value.mean(),summary['mean'],rtol=1e-10,atol=1e-12)
            for metric,summary in sdata['oracle_static_six'].items():
                value=arrays[f'{scenario}_oracle_static_six__{metric}']
                assert np.isclose(value.mean(),summary['mean'],rtol=1e-10,atol=1e-12)
        assert len(all_ids)==len(set(all_ids))
    efficiency=json.loads((HERE/'crn_variance_50node_results.json').read_text(encoding='utf-8'))
    assert efficiency['script_sha256']==sha(HERE/'execution_snapshot/crn_variance_50node_executed.py')
    assert efficiency['path_archive_sha256']==sha(HERE/'crn_variance_50node_paths.npz')
    with np.load(HERE/'crn_variance_50node_paths.npz',allow_pickle=False) as variance_arrays:
        for model,record in efficiency['models'].items():
            assert len(record['replications'])==8
            ratios=[]
            for rep in record['replications']:
                j=rep['replication']
                difference_crn=variance_arrays[f'{model}_crn_negative__mse_rep{j}']-variance_arrays[f'{model}_crn_iid__mse_rep{j}']
                difference_ind=variance_arrays[f'{model}_independent_negative__mse_rep{j}']-variance_arrays[f'{model}_independent_iid__mse_rep{j}']
                assert difference_crn.shape==difference_ind.shape==(512,)
                np.testing.assert_allclose(difference_crn,variance_arrays[f'{model}_crn_difference_rep{j}'],rtol=0,atol=1e-14)
                np.testing.assert_allclose(difference_ind,variance_arrays[f'{model}_independent_difference_rep{j}'],rtol=0,atol=1e-14)
                se_crn=float(difference_crn.std(ddof=1)/np.sqrt(512))
                se_ind=float(difference_ind.std(ddof=1)/np.sqrt(512))
                assert np.isclose(se_crn,rep['paired_path_se_crn'],rtol=1e-12,atol=1e-15)
                assert np.isclose(se_ind,rep['paired_path_se_independent'],rtol=1e-12,atol=1e-15)
                assert se_crn<se_ind
                ratios.append(se_crn/se_ind)
            assert np.isclose(100*(1-np.mean(ratios)),record['mean_relative_se_reduction_percent'])
    seed_groups={'development':set(),'final':set(),'variance':set()}
    for mi,model in enumerate(['unstable','stable']):
        seed_groups['development'].add(dm['seed']+100_000_000+mi*10_000_000)
        seed_groups['final'].update([final['test_seed']+500_000_000+mi*10_000_000,
                                    final['test_seed']+600_000_000+mi*10_000_000])
        seed_groups['variance'].add(efficiency['seed']+1_000_000+mi*100_000)
        for scenario in final['models'][model]['scenarios'].values():
            seed_groups['final'].update(scenario['seed_namespaces'].values())
        for rep in efficiency['models'][model]['replications']:
            seed_groups['variance'].update(rep['seed_namespace_base']+i for i in range(1,7))
    for scenario in dm['scenarios'].values():seed_groups['development'].update(scenario['seed_namespaces'].values())
    assert not(seed_groups['development']&seed_groups['final'])
    assert not(seed_groups['development']&seed_groups['variance'])
    assert not(seed_groups['final']&seed_groups['variance'])
    assert final_analysis['development_selection']=='neg_boundary'
    assert final_analysis['final_scenarios']['stable']['heldout_healthy_fwer']>0.05
    v2=json.loads((ROOT/'paper1_sscd_dev_protocol_20261122/locked_selection.json').read_text(encoding='utf-8'))
    assert v2['selected']=='iid'
    source_registration=json.loads((HERE/'new_experiment_registration.json').read_text(encoding='utf-8'))
    assert source_registration['previous_v2_lock']=='iid, never modified'
    assert source_registration['status'].startswith('Executed:')
    generated_path_ids=[]
    for model in path_manifest['models'].values():
        generated_path_ids += model['calibration_path_ids'] + model['healthy_test_path_ids']
        for scenario in model['scenarios'].values():generated_path_ids += scenario['path_ids']
    assert path_manifest['path_ids_globally_unique_and_disjoint_from_development']
    assert len(generated_path_ids)==len(set(generated_path_ids))
    summary={'status':'PASS','development_scenarios':12,'development_paths_per_scenario':256,
        'final_scenarios':12,'final_paths_per_scenario':512,'final_path_arrays':len(arrays.files),
        'calibration_rank488_verified':True,'path_ids_reconstructed_and_unique':True,
        'development_lock_recomputed_from_saved_paths':True,
        'source_archive_runner_and_development_sources_match':True,'v2_iid_lock_untouched':True,
        'crn_se_reduced_all_16_model_replicates':True,
        'crn_variance_path_arrays':variance_path_count,
        'crn_variance_script_and_archive_hashes_match':True,
        'all_crn_replication_standard_errors_recomputed_from_paths':True,
        'numeric_seed_namespaces_disjoint_between_splits':True,
        'source_package_sha256':original_zip_hash,
        'final_path_manifest_unique':True,
        'stable_healthy_fwer':final_analysis['final_scenarios']['stable']['heldout_healthy_fwer']}
    (HERE/'verification_v3.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
