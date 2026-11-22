"""Validate split integrity, locked choice, hashes and final test read-back."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from selector_v2 import select_from_development, REFERENCE_COMMIT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    development_path = HERE / 'development_losses.json'
    lock_path = HERE / 'locked_selection.json'
    dev_manifest_path = HERE / 'development_manifest.json'
    final_path = HERE / 'final_test_results.json'
    final_manifest_path = HERE / 'final_test_manifest.json'
    development = json.loads(development_path.read_text(encoding='utf-8'))
    development_calibration_path = HERE / 'development_calibration.json'
    development_calibration = json.loads(development_calibration_path.read_text(encoding='utf-8'))
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding='utf-8'))
    final = json.loads(final_path.read_text(encoding='utf-8'))
    final_manifest = json.loads(final_manifest_path.read_text(encoding='utf-8'))
    assert development['split'] == dev_manifest['split'] == 'development'
    assert final['status'] == 'FINAL_TEST_COMPLETE_LOCKED_SELECTION_NOT_REOPENED'
    assert lock['selected'] == final['locked_selection'] == 'iid'
    assert lock['theta_locked'] == 0.0
    assert lock['source_commit'] == dev_manifest['source_commit'] == final['source_commit_reference'] == REFERENCE_COMMIT
    assert sha(development_path) == dev_manifest['development_sha256']
    assert sha(development_calibration_path) == dev_manifest['development_calibration_sha256']
    assert sha(lock_path) == dev_manifest['lock_sha256']
    assert sha(HERE / 'run_development.py') == dev_manifest['development_runner_sha256']
    selector_bytes = (HERE / 'selector_v2.py').read_bytes()
    current_selector_hash = hashlib.sha256(selector_bytes).hexdigest()
    selector_at_run_hash = current_selector_hash
    selector_change_note = 'selector unchanged since development run'
    if current_selector_hash != dev_manifest['development_selector_sha256']:
        historical = HERE / 'execution_snapshot/selector_at_corrected_development_run.py'
        selector_at_run_hash = sha(historical)
        assert selector_at_run_hash == dev_manifest['development_selector_sha256']
        selector_change_note = 'Post-run selector guards were added for global path-ID uniqueness and preventing losses generated at raw theta from being relabeled after MI projection; the exact executed selector snapshot is archived.'
    assert sha(HERE / 'run_final_evaluation.py') == final_manifest['run_final_evaluation_sha256']
    rerun_lock = select_from_development(development, budget=.1)
    assert rerun_lock['selected'] == lock['selected']
    assert rerun_lock['theta_locked'] == lock['theta_locked']

    dev_ids = [pid for scenario in development['scenarios'].values() for pid in scenario['path_ids']]
    assert len(dev_ids) == len(set(dev_ids)) == 12 * 256
    dev_calibration_ids=[]
    for model,record in development_calibration['models'].items():
        assert len(record['path_ids'])==len(record['path_maxima'])==512
        assert len(set(record['path_ids']))==512
        assert np.isclose(np.partition(record['path_maxima'],487)[487],
                          dev_manifest[f'{model}_threshold']['threshold'],rtol=0,atol=1e-12)
        dev_calibration_ids.extend(record['path_ids'])
    assert len(dev_calibration_ids)==len(set(dev_calibration_ids))==1024
    assert not (set(dev_ids)&set(dev_calibration_ids))
    assert len(development['scenarios']) == 12
    for name, rec in development['scenarios'].items():
        assert len(rec['path_ids']) == 256
        assert set(rec['losses']) == {'neg_boundary', 'neg06', 'iid', 'pos06', 'pos_boundary'}
        assert set(rec['affected_losses']) == set(rec['losses'])
        assert all(len(v) == 256 for v in rec['losses'].values())
        assert all(len(v) == 256 for v in rec['affected_losses'].values())

    test_ids, check_count = [], 0
    arrays = np.load(HERE / 'final_test_paths.npz', allow_pickle=False)
    dev_id_set = set(dev_ids)
    for model in ['unstable', 'stable']:
        record = final['models'][model]
        assert record['threshold']['calibration_paths'] == 512
        assert record['independent_healthy_test']['paths'] == 512
        assert len(record['scenarios']) == 6
        calibration_ids = record['calibration_path_ids']
        healthy_ids = record['independent_healthy_test']['path_ids']
        assert len(calibration_ids) == len(set(calibration_ids)) == 512
        assert len(healthy_ids) == len(set(healthy_ids)) == 512
        test_ids.extend(calibration_ids)
        test_ids.extend(healthy_ids)
        assert not (set(calibration_ids) & dev_id_set)
        assert not (set(healthy_ids) & dev_id_set)
        assert not (set(calibration_ids) & set(dev_calibration_ids))
        assert not (set(healthy_ids) & set(dev_calibration_ids))
        calibration_maxima = arrays[f'{model}_calibration_path_max']
        assert calibration_maxima.shape == (512,)
        assert np.isclose(np.partition(calibration_maxima, 487)[487],
                          record['threshold']['threshold'], rtol=0, atol=1e-12)
        healthy_alarm = arrays[f'{model}_healthy_test__path_any_alarm']
        assert int(healthy_alarm.sum()) == record['independent_healthy_test']['path_any_alarm_count']
        for scenario, s in record['scenarios'].items():
            assert len(s['path_ids']) == 512 and len(set(s['path_ids'])) == 512
            test_ids.extend(s['path_ids'])
            assert not (set(s['path_ids']) & dev_id_set)
            assert set(s['candidates']) == {'neg_boundary', 'neg06', 'iid', 'pos06', 'pos_boundary'}
            for policy in s['candidates']:
                for key, item in s['candidates'][policy].items():
                    assert item['paths'] == 512 and np.isfinite(item['mean'])
                    arr = arrays[f'{scenario}_{policy}__{key}']
                    assert arr.shape == (512,) and np.all(np.isfinite(arr))
                check_count += 1
            for key, item in s['oracle_static_six'].items():
                assert item['paths'] == 512 and np.isfinite(item['mean'])
                assert arrays[f'{scenario}_oracle_static_six__{key}'].shape == (512,)
    assert len(test_ids) == len(set(test_ids)) == 2 * (512 + 512) + 12 * 512
    assert not (set(test_ids) & dev_id_set)
    assert not (set(test_ids)&set(dev_calibration_ids))
    assert final_manifest['scenario_count'] == 12
    assert final_manifest['test_paths_per_scenario_and_model'] == 512
    for name, expected in dev_manifest['fixture_sha256'].items():
        path = (ROOT / 'paper1_50node_tv_gate_theory_20261121' / name
                if name == 'replay.py' else ROOT / 'paper1_common_g50_gain_acceptance_20261121' / name)
        assert sha(path) == expected
    report = {
        'status': 'PASS', 'locked_selection': 'iid',
        'development_scenarios': 12, 'development_paths_per_scenario': 256,
        'development_path_ids_globally_unique': True,
        'development_calibration_path_maxima_and_ids_saved_rank488_verified': True,
        'development_calibration_ids_disjoint_from_development_fit_ids': True,
        'final_calibration_paths_per_model': 512,
        'final_test_scenarios': 12, 'final_test_paths_per_scenario': 512,
        'final_test_path_ids_globally_unique_and_disjoint_from_development_calibration_and_fit': True,
        'locked_choice_recomputed_from_unchanged_development_data': True,
        'selector_hash_at_development_run': selector_at_run_hash,
        'selector_hash_current': current_selector_hash,
        'selector_change_note': selector_change_note,
        'candidate_world_records_checked': check_count,
        'path_level_final_arrays': len(arrays.files),
        'frozen_source_hashes_match': True,
    }
    (HERE / 'verification_v2.json').write_text(json.dumps(report, indent=2) + '\n',
                                               encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
