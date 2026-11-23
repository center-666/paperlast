"""Recompute registered C1/C2 paired summaries from the stored path arrays."""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.stats import t

from ablation import HERE, BASE, read, summary


def paired(left, right, bonferroni_m=1):
    x = np.asarray(left, float)-np.asarray(right, float)
    se = float(x.std(ddof=1)/np.sqrt(len(x)))
    degrees = len(x)-1
    crit = float(t.ppf(1-.05/bonferroni_m, degrees))
    return {**summary(x), 'ci95_two_sided':[
        float(x.mean()-t.ppf(.975,degrees)*se),
        float(x.mean()+t.ppf(.975,degrees)*se)],
        'one_sided_bonferroni_upper':float(x.mean()+crit*se),
        'bonferroni_comparisons':bonferroni_m}


def c1_report(results, archive):
    report = {'calibration_fwer':{}, 'primary_negative_watermark':{}, 'all_watermarks':{}}
    names=('H4','H2','H1','EWMA-A')
    for model, data in results['models'].items():
        report['calibration_fwer'][model] = data['calibration']['health_fwer']
        report['all_watermarks'][model]={}
        report['primary_negative_watermark'][model]={}
        for fixture,record in data['fixtures'].items():
            report['all_watermarks'][model][fixture]={}
            for watermark in ('iid','negative','positive'):
                metrics={}
                prefix=f'C1.{model}.{fixture}.{watermark}.'
                for name in names:
                    metrics[name]={}
                    for metric in ('gap_reacq','gap_censored','gap_accept_fraction',
                                   'fault_alarm_fraction','detection_delay','any_fault_alarm',
                                   'missing_fault_window','missing_after_fault','longest_run',
                                   'longest_run_window','seven_contract'):
                        key=prefix+name+'.'+metric
                        if key in archive:
                            metrics[name][metric]=summary(archive[key].astype(float))
                comparisons={}
                for name in names[1:]:
                    comparisons[name]={}
                    for metric in ('gap_reacq','fault_alarm_fraction','missing_fault_window','missing_after_fault'):
                        key=prefix+name+'.'+metric
                        if key in archive:
                            comparisons[name][metric]=paired(archive[key],archive[prefix+'H4.'+metric],bonferroni_m=3)
                row={'detectors':metrics,'paired_minus_H4':comparisons}
                report['all_watermarks'][model][fixture][watermark]=row
                if watermark=='negative':
                    report['primary_negative_watermark'][model][fixture]=row
    return report


def c2_report(results, archive):
    report={'calibration_fwer':{},'primary_negative_watermark':{},'method_contrasts':{}}
    for model,data in results['models'].items():
        report['calibration_fwer'][model]={
            'health_fwer':data['calibration']['health_fwer'],
            'gate_thresholds':data['calibration']['gate_thresholds'],
            'trusted_H1_threshold':data['calibration']['trusted_reference_threshold'],
            'pathmax_quantiles':{name:np.quantile(archive[f'C2.{model}.calibration.{name}.maxima'],
                    [.5,.9,.95,.99,1.]).tolist() for name in ('H4','EWMA-A','EWMA-B','EWMA-C')}}
        report['primary_negative_watermark'][model]={}
        report['method_contrasts'][model]={}
        for fixture,record in data['fixtures'].items():
            report['primary_negative_watermark'][model][fixture]={}
            report['method_contrasts'][model][fixture]={}
            for detector_key,values in record['detectors'].items():
                name=detector_key.rsplit('.',1)[-1]
                key=detector_key+'.'
                selected=('fault_alarm_fraction','detection_delay','any_fault_alarm','stop_reacq',
                          'stop_censored','gap_reacq','gap_censored','gap_accept_fraction',
                          'longest_run','longest_run_window','seven_contract',
                          'missing_fault_window','missing_after_fault','missing_post20',
                          'inherited_state','subsequent_gate')
                report['primary_negative_watermark'][model][fixture][name]={
                    metric:summary(archive[key+metric].astype(float)) for metric in selected
                    if key+metric in archive}
                if name in ('EWMA-B','EWMA-C'):
                    bits=archive[key+'update_bits']
                    updates=(bits & (1<<6))>0
                    active=np.asarray(record['active_mask'],bool)
                    report['primary_negative_watermark'][model][fixture][name]['trusted_update_fraction_fault']=float(
                        updates[:,active].mean())
                    report['primary_negative_watermark'][model][fixture][name]['trusted_update_fraction_all']=float(
                        updates[:,4:].mean())
                    fault_updates=updates[:,active]
                    first=np.where(fault_updates.any(axis=1),np.argmax(fault_updates,axis=1),fault_updates.shape[1])
                    report['primary_negative_watermark'][model][fixture][name]['paths_with_any_trusted_update']=float(fault_updates.any(axis=1).mean())
                    report['primary_negative_watermark'][model][fixture][name]['first_trusted_update_delay_from_onset']=summary(first.astype(float))
            methods=report['primary_negative_watermark'][model][fixture]
            pA=f'C2.{model}.{fixture}.negative.EWMA-A.'
            for method in ('EWMA-B','EWMA-C'):
                pM=f'C2.{model}.{fixture}.negative.{method}.'
                report['method_contrasts'][model][fixture][f'{method}_minus_EWMA-A']={
                    metric:paired(archive[pM+metric],archive[pA+metric],bonferroni_m=2)
                    for metric in ('fault_alarm_fraction','detection_delay','gap_reacq',
                                   'gap_accept_fraction','missing_fault_window','missing_after_fault')
                    if pM+metric in archive and pA+metric in archive}
            report['method_contrasts'][model][fixture]['EWMA-C_minus_EWMA-B']={
                metric:paired(archive[f'C2.{model}.{fixture}.negative.EWMA-C.'+metric],
                              archive[f'C2.{model}.{fixture}.negative.EWMA-B.'+metric],bonferroni_m=1)
                for metric in ('fault_alarm_fraction','detection_delay','gap_reacq',
                               'gap_accept_fraction','missing_fault_window','missing_after_fault')
                if f'C2.{model}.{fixture}.negative.EWMA-C.'+metric in archive and
                   f'C2.{model}.{fixture}.negative.EWMA-B.'+metric in archive}
    return report


def main():
    result={'status':'PASS','scope':'Descriptive analysis of registered fixed detector/watermark comparisons.',
            'independent_new_C1_C2_samples':True,'C2_score_caveat':
            'EWMA-A is inclusive plug-in KL. EWMA-B/C use prequential Mahalanobis energy. A/B differ in score family and update ordering, so the contrast does not isolate self-inclusion alone.'}
    for phase in ('C1','C2'):
        folder=HERE/'results'/phase
        report=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        if phase=='C1':
            path=folder/'paths.npz'
            assert report['archive_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as raw:
                arrays={name:raw[name].copy() for name in raw.files}
        else:
            storage=read(folder/'archive_manifest.json')
            assert report['storage_archive']['original_execution_archive_sha256']==storage['original_execution_archive_sha256']
            arrays={}
            for shard,entry in storage['shards'].items():
                path=folder/shard
                assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
                with np.load(path,allow_pickle=False) as raw:
                    arrays.update({name:raw[name].copy() for name in raw.files})
        assert len(arrays)==report['array_count']
        result[phase]=(c1_report(report,arrays) if phase=='C1' else c2_report(report,arrays))
    output=HERE/'analysis_c.json'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','C1_models':list(result['C1']['all_watermarks']),
                      'C2_models':list(result['C2']['primary_negative_watermark']),
                      'output':str(output)},indent=2))


if __name__=='__main__':main()
