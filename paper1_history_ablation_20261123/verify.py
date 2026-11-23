"""Read back calibration ranks, gate decisions, path comparisons and tail effects."""

import hashlib
import subprocess
from contextlib import ExitStack
import numpy as np
from scipy.stats import t
from ablation import HERE, ROOT, BASE, NAMES, EDGE, START, STOP, read, digest, summary, scores_all


def summary_check(x, expected):
    actual = summary(x.astype(float))
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-12)


def weighted_reference():
    # Check online EWMA against normalized explicit weights, including startup.
    rng = np.random.default_rng(109)
    residual = rng.normal(size=(4, 146, 2, 8))
    ref = np.array([np.eye(8), 1.7*np.eye(8)])
    series = scores_all(residual, ref)
    for j in [1, 3, 11, 145]:
        for edge in range(2):
            v = residual[0,:j+1,edge]
            weights = .5**np.arange(j,-1,-1)
            weights /= weights.sum()
            mean = weights @ v
            cov = np.cov(v.T, aweights=weights, ddof=1) + 1e-6*np.eye(8)
            score = .5*(np.trace(np.linalg.solve(ref[edge],cov)) + mean @ np.linalg.solve(ref[edge],mean)
                        -8 + np.linalg.slogdet(ref[edge])[1]-np.linalg.slogdet(cov)[1])
            np.testing.assert_allclose(series['EWMA'][0,j+4,edge],score,atol=1e-7)
            np.testing.assert_allclose(series['H1'][0,j+4,edge],.5*v[-1] @ np.linalg.solve(ref[edge],v[-1]))


def first_or_limit(events):
    return np.where(events.any(axis=-1), np.argmax(events,axis=-1), events.shape[-1])


def main():
    weighted_reference()
    result = read(HERE/'results.json')
    assert digest(HERE/'ablation.py') == result['runner_sha256']
    assert digest(HERE/'REGISTRATION.md') == result['registration_sha256']
    committed = subprocess.check_output(['git','show','9acfaca:paper1_history_ablation_20261123/REGISTRATION.md'],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == result['registration_sha256']
    assert digest(HERE/'paths.npz') == result['archive_sha256']
    source = read(BASE/'critical_results.json')
    exp_a = ROOT/'paper1_critical_gate_crossover_20261123'
    assert digest(exp_a/'crossover.npz') == read(exp_a/'crossover.json')['path_archive_sha256']
    total = 0
    with ExitStack() as files:
        arrays = files.enter_context(np.load(HERE/'paths.npz', allow_pickle=False))
        historic = files.enter_context(np.load(BASE/'critical_paths.npz', allow_pickle=False))
        previous = files.enter_context(np.load(exp_a/'crossover.npz', allow_pickle=False))
        assert result['array_count'] == len(arrays.files)
        for label, model in result['models'].items():
            for i,name in enumerate(NAMES):
                record = model['calibration'][name]
                quantiles = np.sort(arrays[label+'_cal_maxima'][:,i],axis=-1)[:,487]
                np.testing.assert_array_equal(quantiles,record['threshold_by_batch'])
                hold = arrays[label+'_hold_maxima'][:,i].ravel()
                assert [(hold > x).sum() for x in quantiles] == record['holdout_counts_by_batch']
            for fixture, policies in model['fixtures'].items():
                mask = np.asarray(source['models'][label]['fixtures'][fixture]['fixture_active_mask'])
                last = np.where(mask)[0][-1]+1
                for policy, detectors in policies.items():
                    prefix = f'{label}.{fixture}.{policy}.'
                    for name in NAMES:
                        key = prefix+name+'.'
                        saved = detectors[name]
                        bits = arrays[key+'bits']
                        assert bits.shape == (512,150) and bits.max()<128
                        g = (bits & (1<<EDGE)) != 0
                        clean = arrays[f'{label}.{fixture}.clean.{name}.bits']
                        np.testing.assert_array_equal(bits & 63, clean & 63)
                        if name=='H4':
                            np.testing.assert_array_equal(g,historic[f'{label}__{fixture}__{policy}__critical_gate'])
                            prior_gates = previous[f'{label}__{fixture}__{policy}__frozen_gates']
                            np.testing.assert_array_equal(bits, np.packbits(prior_gates, axis=2, bitorder='little')[:,:,0])
                        if name!='EWMA':
                            cutoff = last+3+{'H4':4,'H2':2,'H1':1}[name]
                            np.testing.assert_array_equal(bits[:,cutoff:],clean[:,cutoff:])
                        np.testing.assert_array_equal(first_or_limit(~g[:,last:]),arrays[key+'stop_reacq'])
                        np.testing.assert_array_equal(g[:,mask].mean(axis=1),arrays[key+'fault_alarm_fraction'])
                        np.testing.assert_array_equal(first_or_limit(g[:,START:STOP]&mask[START:STOP]),
                                                      arrays[key+'detection_delay'])
                        if fixture=='burst':
                            gaps = g[:,START:STOP].reshape(512,6,10)[:,:,6:]
                            duration = first_or_limit(~gaps)
                            np.testing.assert_array_equal(duration,arrays[key+'gap_durations'])
                            np.testing.assert_array_equal(duration.mean(axis=1),arrays[key+'gap_reacq'])
                            np.testing.assert_array_equal((duration==4).mean(axis=1),arrays[key+'gap_censored'])
                            if name!='H4':
                                diff = duration.mean(axis=1)-arrays[prefix+'H4.gap_reacq']
                                expected = diff.mean()+t.ppf(1-.05/3,511)*diff.std(ddof=1)/np.sqrt(512)
                                np.testing.assert_allclose(expected,detectors['paired_vs_H4'][name]['gap_one_sided_bonferroni_upper'])
                        aa,ac,ca,cc=(arrays[key+'tail_'+s] for s in ('aa','ac','ca','cc'))
                        np.testing.assert_allclose(arrays[key+'inherited_state'],((ac-cc)+(aa-ca))/2)
                        np.testing.assert_allclose(arrays[key+'subsequent_gate'],((ca-cc)+(aa-ac))/2)
                        for metric, expected in saved.items():
                            x = arrays[key+metric]
                            summary_check(x,expected)
                            if name!='H4':
                                paired = x.astype(float)-arrays[prefix+'H4.'+metric].astype(float)
                                summary_check(paired,detectors['paired_vs_H4'][name][metric])
                            total+=1
    print(f'PASS: {result["array_count"]} arrays, {total} primary/secondary path summaries, paired contrasts, threshold ranks, EWMA independent check, tail decompositions.')


if __name__=='__main__':
    main()
