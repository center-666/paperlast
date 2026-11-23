"""Read-back audit for the fresh v3 CRN development/final runs."""
from pathlib import Path
import hashlib
import json
from zipfile import ZipFile
import numpy as np
from scipy.stats import t as student_t

HERE=Path(__file__).resolve().parent


def stratified_pair(arrays, scenarios, candidate, metric, n_paths):
    differences=[]
    for scenario in scenarios:
        diff=arrays[f'{scenario}_{candidate}__{metric}']-arrays[f'{scenario}_iid__{metric}']
        assert diff.shape==(n_paths,) and np.isfinite(diff).all()
        differences.append(diff)
    count=len(differences)
    means=np.array([x.mean() for x in differences])
    components=[x.var(ddof=1)/len(x)/(count**2) for x in differences]
    variance=float(sum(components));se=float(np.sqrt(variance))
    denom=sum(v*v/(len(x)-1) for v,x in zip(components,differences))
    df=variance**2/denom if denom>0 else 1e9
    q=float(student_t.ppf(.975,df))
    return {'equal_weight_scenario_mean':float(means.mean()),'stratified_se':se,
            'satterthwaite_df':float(df),'conditional_unadjusted_95ci':[float(means.mean()-q*se),float(means.mean()+q*se)],
            'scenarios_with_negative_mean':int((means<0).sum()),'scenario_count':count,
            'paths_per_scenario':n_paths,'scenario_means':means.tolist()}


def main():
    development=json.loads((HERE/'crn_development_losses.json').read_text(encoding='utf-8'))
    devmanifest=json.loads((HERE/'crn_development_manifest.json').read_text(encoding='utf-8'))
    lock=json.loads((HERE/'crn_locked_selection.json').read_text(encoding='utf-8'))
    final=json.loads((HERE/'crn_final_results.json').read_text(encoding='utf-8'))
    arrays=np.load(HERE/'crn_final_path_arrays.npz',allow_pickle=False)
    assert development['split']=='development_crn'
    assert lock['status']=='FRESH_DEVELOPMENT_DESIGN_ONLY_NOT_A_TEST'
    assert final['status']=='INDEPENDENT_TEST_EXECUTED_NO_RESELECTION'
    assert final['new_v3_selection']==lock['selection']
    assert hashlib.sha256((HERE/'crn_development_losses.json').read_bytes()).hexdigest()==devmanifest['development_sha256']
    assert hashlib.sha256((HERE/'crn_locked_selection.json').read_bytes()).hexdigest()==devmanifest['lock_sha256']==final['lock_sha256']
    assert hashlib.sha256((HERE/'crn_final_path_arrays.npz').read_bytes()).hexdigest()==final['path_archive_sha256']
    with ZipFile(HERE/'source_package.zip') as package_zip:
        packaged_final_runner=package_zip.read('paper1_sscd_commonrandom_20261122/run_crn_final.py')
    packaged_final_runner_hash=hashlib.sha256(packaged_final_runner).hexdigest()
    assert packaged_final_runner_hash==final['run_script_sha256']
    development_path_ids=[]
    for scenario in development['scenarios'].values():
        development_path_ids.extend(scenario['path_ids'])
    assert len(development_path_ids)==len(set(development_path_ids))
    report={'status':'PASS','development_selection':lock['selection'],'theta':lock['theta_locked'],
            'v3_registration':'epsilon/gamma thresholds were proposed after review of old v2 results; this is a new exploratory v3 split, not retroactive confirmation.',
            'development_scenarios':len(development['scenarios']),'final_scenarios':{},
            'crn_efficiency_50node':json.loads((HERE/'crn_variance_50node_results.json').read_text(encoding='utf-8'))['models'],
            'high_dim_sampler_endpoint_validation':json.loads((HERE/'nd_endpoint_validation.json').read_text(encoding='utf-8'))}
    for model,record in final['models'].items():
        scenarios=list(record['scenarios'])
        n_paths=final['test_paths_per_scenario']
        candidate_means={}
        for policy in ['iid','neg_boundary','neg06','pos06','pos_boundary']:
            candidate_means[policy]={}
            for metric in ['network_post20','affected_post20','network_attack','network_postattack',
                           'eligible_link_time_alarm_fraction','eligible_first_alarm_delay_from_onset']:
                candidate_means[policy][metric]=float(np.mean([
                    record['scenarios'][s]['candidate_means'][policy][metric]['mean'] for s in scenarios]))
        paired={}
        for policy in ['neg_boundary','neg06','pos06','pos_boundary']:
            paired[policy]={metric:stratified_pair(arrays,scenarios,policy,metric,n_paths)
                            for metric in ['network_post20','affected_post20','network_attack',
                                           'eligible_link_time_alarm_fraction','eligible_first_alarm_delay_from_onset']}
        oracle={metric:float(np.mean([record['scenarios'][s]['oracle_static_six'][metric]['mean'] for s in scenarios]))
                for metric in ['network_post20','affected_post20','network_attack','network_postattack']}
        report['final_scenarios'][model]={
            'threshold':record['calibration_threshold'],'heldout_healthy_fwer':record['heldout_healthy_FWER'],
            'healthy_alarms':record['healthy_alarms'],'healthy_fwer_exact_binomial_ci95':record['healthy_FWER_ClopperPearson_CI95'],
            'candidate_equal_weight_means':candidate_means,'paired_candidate_minus_iid':paired,
            'known_six_static_oracle_equal_weight_means':oracle,
            'selected_candidate_vs_fixed_negative':'The v3 selector selected the boundary-negative candidate, equal to the previous fixed negative theta after MI-feasibility correction; this does not establish an optimizer advantage over that fixed candidate.'}
    report['provenance']={'development_manifest_sha256':hashlib.sha256((HERE/'crn_development_manifest.json').read_bytes()).hexdigest(),
                          'lock_sha256':final['lock_sha256'],'final_runner_sha256_recorded':final['run_script_sha256'],
                          'final_runner_sha256_in_source_zip':packaged_final_runner_hash,
                          'path_array_count':len(arrays.files)}
    assert report['provenance']['final_runner_sha256_recorded']==report['provenance']['final_runner_sha256_in_source_zip']
    final_seed=int(final['test_seed']);path_manifest={'status':'PASS','path_ids_source':'Deterministically derived from the exact recorded RNG seed namespaces; original final runner saved path arrays indexed 0..N-1 but did not include string IDs.',
        'final_seed':final_seed,'models':{}}
    for mi,model in enumerate(['unstable','stable']):
        calibration_seed=final_seed+500_000_000+mi*10_000_000
        healthy_seed=final_seed+600_000_000+mi*10_000_000
        model_manifest={'calibration_seed':calibration_seed,
            'calibration_path_ids':[f'crn_final_cal_{calibration_seed}_{model}_{i}' for i in range(final['calibration_paths_per_model'])],
            'healthy_test_seed':healthy_seed,
            'healthy_test_path_ids':[f'crn_final_healthy_{healthy_seed}_{model}_{i}' for i in range(final['test_paths_per_scenario'])],
            'scenarios':{}}
        for ai,angle in enumerate([0.,-45.,45.]):
            for oi,onset in enumerate([50,65]):
                ix=ai*2+oi;scenario=f'{model}__rot{angle:+.0f}__onset{onset}'
                seeds={
                    'physical':final_seed+700_000_000+1_000_000+(mi*6+ix)*1_000_000,
                    'radial':final_seed+700_000_000+20_000_000+(mi*6+ix)*1_000_000,
                    'initial':final_seed+700_000_000+40_000_000+(mi*6+ix)*1_000_000,
                    'quantile_and_sign':final_seed+700_000_000+60_000_000+(mi*6+ix)*1_000_000}
                assert final['models'][model]['scenarios'][scenario]['seed_namespaces']==seeds
                model_manifest['scenarios'][scenario]={'seed_namespaces':seeds,
                    'path_ids':[f'crn_final_{seeds["physical"]}_{scenario}_{i}' for i in range(final['test_paths_per_scenario'])]}
        path_manifest['models'][model]=model_manifest
    all_final_ids=[]
    for model in path_manifest['models'].values():
        all_final_ids.extend(model['calibration_path_ids'])
        all_final_ids.extend(model['healthy_test_path_ids'])
        for scenario in model['scenarios'].values():all_final_ids.extend(scenario['path_ids'])
    assert len(all_final_ids)==len(set(all_final_ids))
    assert not(set(all_final_ids)&set(development_path_ids))
    path_manifest['path_ids_globally_unique_and_disjoint_from_development']=True
    (HERE/'crn_final_path_manifest.json').write_text(json.dumps(path_manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (HERE/'v3_final_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    for model,data in report['final_scenarios'].items():
        print(model,'threshold',data['threshold'],'healthy FWER',data['heldout_healthy_fwer'],
              'CRN boundary MSE diff',data['paired_candidate_minus_iid']['neg_boundary']['network_post20']['equal_weight_scenario_mean'],
              'static oracle',data['known_six_static_oracle_equal_weight_means']['network_post20'])
    print('analysis output:',HERE/'v3_final_analysis.json')


if __name__=='__main__':main()
