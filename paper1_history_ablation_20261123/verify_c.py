"""Verify registered C1/C2 sources, random streams, thresholds and path summaries."""

from contextlib import ExitStack
import hashlib
import json
import subprocess

import numpy as np

from ablation import HERE, BASE
from run_critical import digest
from experiment_c import C1_NAMES, C2_NAMES, ROOT1, ROOT2, N_CAL, N_HOLD, N_TEST, RANK, RECOVERY_RUN


def load_phase_arrays(phase, folder, result):
    if phase == 'C1':
        path = folder/'paths.npz'
        assert digest(path) == result['archive_sha256']
        with np.load(path, allow_pickle=False) as source:
            arrays = {name:source[name].copy() for name in source.files}
        return arrays, digest(path)

    storage = json.loads((folder/'archive_manifest.json').read_text(encoding='utf-8'))
    assert result['storage_archive']['original_execution_archive_sha256'] == result['archive_sha256']
    arrays = {}
    for shard, info in storage['shards'].items():
        path = folder/shard
        assert path.stat().st_size < 100_000_000
        assert digest(path) == info['sha256']
        with np.load(path, allow_pickle=False) as source:
            arrays.update({name:source[name].copy() for name in source.files})
    assert len(arrays) == result['array_count'] == len(storage['arrays'])
    for name, value in arrays.items():
        info = storage['arrays'][name]
        assert list(value.shape) == info['shape'] and value.dtype.str == info['dtype']
        checksum = hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        assert checksum == info['array_sha256'], name
    return arrays, storage['original_execution_archive_sha256']


def verify_runner_registration(phase, result, folder):
    if phase == 'C1':
        runner = HERE/'experiment_c_c1_executed.py'
        registration = subprocess.check_output(
            ['git', 'show', 'bb2cc79:paper1_history_ablation_20261123/REGISTRATION_C1_C2_AMENDMENT.md'],
            cwd=HERE.parent)
        assert digest(runner) == result['runner_sha256']
        assert hashlib.sha256(registration).hexdigest() == result['registration_amendment_sha256']
        execution = {
            'phase': 'C1', 'execution_source_commit': 'bb2cc79',
            'runner_sha256': result['runner_sha256'],
            'registration_sha256': result['registration_amendment_sha256'],
            'results_sha256': digest(folder/'results.json'),
            'root_seed': ROOT1, 'calibration_paths_per_model': N_CAL,
            'independent_holdout_paths_per_model': N_HOLD,
            'attack_paths_per_model_and_fixture': N_TEST,
            'executed_source_snapshot': runner.name,
        }
    else:
        runner = HERE/'experiment_c_c2_executed.py'
        assert digest(runner) == result['runner_sha256']
        assert result['registration_amendment_sha256'] == digest(HERE/'REGISTRATION_C1_C2_AMENDMENT.md')
        startup_bytes = subprocess.check_output(
            ['git', 'show', 'eadc22a:paper1_history_ablation_20261123/REGISTRATION_C2_STARTUP.md'],
            cwd=HERE.parent)
        assert hashlib.sha256(startup_bytes).hexdigest() == digest(HERE/'REGISTRATION_C2_STARTUP.md')
        code = subprocess.check_output(
            ['git', 'show', 'eadc22a:paper1_history_ablation_20261123/experiment_c.py'], cwd=HERE.parent)
        assert hashlib.sha256(code).hexdigest() == result['runner_sha256']
        execution = json.loads((folder/'execution_manifest.json').read_text(encoding='utf-8'))
        assert execution['runner_sha256'] == result['runner_sha256']
        execution.update({
            'execution_source_commit': 'eadc22a',
            'startup_registration_sha256': digest(HERE/'REGISTRATION_C2_STARTUP.md'),
            'results_sha256': digest(folder/'results.json'),
        })
    return execution


def main():
    stage_summaries = {}
    all_seeds = {'C1':set(), 'C2':set()}
    executions = {}
    for phase, root, methods in (('C1', ROOT1, C1_NAMES), ('C2', ROOT2, C2_NAMES)):
        folder = HERE/'results'/phase
        result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert result['phase'] == phase
        arrays, archive_sha = load_phase_arrays(phase, folder, result)
        executions[phase] = verify_runner_registration(phase, result, folder)
        assert result['previous_critical_archive_sha256'] == digest(BASE/'critical_paths.npz')
        verified_summaries = 0

        for model_index, model in enumerate(('unstable', 'stable')):
            record = result['models'][model]
            calibration = record['calibration']
            gate_seed = root + 100000 + 10000*model_index
            holdout_seed = root + (200000 if phase == 'C1' else 300000) + 10000*model_index
            assert calibration['gate_seed'] == gate_seed
            assert calibration['holdout_seed'] == holdout_seed
            all_seeds[phase].update((gate_seed, holdout_seed))

            if phase == 'C2':
                trusted_seed = root + 200000 + 10000*model_index
                assert calibration['trusted_reference_seed'] == trusted_seed
                all_seeds[phase].add(trusted_seed)
                trusted_maxima = arrays[f'{phase}.{model}.trust_reference.maxima']
                assert len(trusted_maxima) == N_CAL
                np.testing.assert_equal(np.partition(trusted_maxima, RANK-1)[RANK-1],
                                        calibration['trusted_reference_threshold'])

            for method in methods:
                maxima = arrays[f'{phase}.{model}.calibration.{method}.maxima']
                assert len(maxima) == N_CAL
                rank = calibration['rank_one_based']
                np.testing.assert_equal(np.partition(maxima, rank-1)[rank-1],
                                        calibration['gate_thresholds'][method])
                alarms = arrays[f'{phase}.{model}.holdout.{method}.alarm']
                assert len(alarms) == N_HOLD
                assert int(alarms.sum()) == calibration['health_fwer'][method]['k']
            for fixture_index, (fixture, fixture_record) in enumerate(record['fixtures'].items()):
                attack_seed = root + (300000 if phase == 'C1' else 400000) \
                    + 10000*model_index + 1000*fixture_index
                assert fixture_record['physical_seed'] == attack_seed
                all_seeds[phase].update((attack_seed, attack_seed+11, attack_seed+12, attack_seed+13))
                for key, expected in fixture_record['detectors'].items():
                    masks = arrays[key+'.gate_bits']
                    assert masks.shape == (N_TEST,150) and int(masks.max()) < 128
                    if phase == 'C2' and key.endswith(('EWMA-B','EWMA-C')):
                        assert np.all(arrays[key+'.critical_score'][:,4] == 0)
                    for metric, stats in expected.items():
                        values = arrays[key+'.'+metric]
                        assert values.shape == (N_TEST,) and np.isfinite(values).all()
                        mean = float(values.mean())
                        sem = float(values.std(ddof=1)/np.sqrt(N_TEST))
                        assert np.isclose(mean,stats['mean'],rtol=1e-10,atol=1e-11)
                        assert np.isclose(sem,stats['path_se'],rtol=1e-9,atol=1e-11)
                        verified_summaries += 1
                    if phase == 'C2' and key.endswith('.EWMA-B'):
                        bits = arrays[key+'.update_bits']
                        assert bits.shape == (N_TEST,150)
                        update_mask=(bits & (1<<6)) != 0
                        assert not update_mask[:,:4].any() and update_mask[:,4:].all()
                    if phase == 'C2' and key.endswith('.EWMA-C'):
                        bits = arrays[key+'.update_bits']
                        trusted_scores = arrays[key+'.trusted_score_critical']
                        assert bits.shape == trusted_scores.shape == (N_TEST,150)
                        trusted = trusted_scores <= calibration['trusted_reference_threshold']
                        expected_updates = np.zeros_like(trusted)
                        streak = np.zeros(N_TEST,np.int16)
                        for tick in range(4,150):
                            streak = np.where(trusted[:,tick],np.minimum(streak+1,RECOVERY_RUN),0)
                            expected_updates[:,tick] = trusted[:,tick] & (streak >= RECOVERY_RUN)
                        np.testing.assert_array_equal((bits & (1<<6)) != 0, expected_updates)
                        for name in ('mean_critical','covariance_diagonal_critical'):
                            assert np.isfinite(arrays[key+'.'+name]).all()

        stage_summaries[phase] = {
            'results_sha256': digest(folder/'results.json'),
            'source_execution_archive_sha256': result['archive_sha256'],
            'stored_archive_sha256': archive_sha,
            'array_count': result['array_count'],
            'verified_path_summaries': verified_summaries,
            'seed_root': root,
        }

    assert not (all_seeds['C1'] & all_seeds['C2'])
    for phase in ('C1','C2'):
        path = HERE/'results'/phase/'execution_manifest.json'
        path.write_text(json.dumps(executions[phase],indent=2)+'\n',encoding='utf-8',newline='\n')
        verified = HERE/'results'/phase/'verified_manifest.json'
        verified.write_text(json.dumps(stage_summaries[phase],indent=2)+'\n',encoding='utf-8',newline='\n')
    summary = {'status':'PASS','stages':stage_summaries,'seed_namespaces_disjoint':True,
               'EWMA_C_five_trusted_sample_rule_verified':True}
    print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
