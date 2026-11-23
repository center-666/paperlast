"""Verify the stored results and the fixed-schedule Gaussian error moments."""

import json

import numpy as np

from run_critical import HERE, ROOT, PHYSICAL, RECEIVER, START, STOP, T, critical_model, digest, summary


def check_summary(values, record):
    assert values.shape == (record['n'],) and np.isfinite(values).all()
    computed = summary(values.astype(float))
    for key in computed:
        np.testing.assert_allclose(computed[key], record[key], atol=1e-12, rtol=1e-12)


def longest_run(gates):
    return np.array([max((len(run) for run in ''.join('1' if x else '0' for x in row).split('0')),
                         default=0) for row in gates])


def timing_oracle_covariance(model, reject):
    neighbors = np.array(model.cfg['graph']['neighbors'][RECEIVER])
    retained = neighbors[neighbors != 35]
    covariance = np.eye(4)
    moments = []
    for k in range(1, T):
        mask = 64 if reject[k] else 0
        chosen = retained if reject[k] else neighbors
        K, G = model.K[mask, RECEIVER], model.G[mask, RECEIVER]
        correction = np.eye(4) - K @ model.H[RECEIVER] - G @ model.H[chosen].sum(axis=0)
        predicted = model.A @ covariance @ model.A.T + model.Q
        covariance = (correction @ predicted @ correction.T + K @ model.R[RECEIVER] @ K.T
                      + G @ model.R[chosen].sum(axis=0) @ G.T)
        moments.append(np.trace(covariance[2:, 2:]))
    return np.array(moments)


def main():
    result = json.loads((HERE / 'critical_results.json').read_text(encoding='utf-8'))
    cfg = json.loads(PHYSICAL.read_text(encoding='utf-8'))
    assert result['registration_sha256'] == digest(HERE / 'registration.json')
    assert result['script_sha256'] == digest(HERE / 'run_critical.py')
    for path, expected in result['sources'].items():
        assert digest(ROOT / path) == expected, path
    archive = HERE / 'critical_paths.npz'
    assert result['path_archive_sha256'] == digest(archive)
    audit = {'status': 'PASS', 'fixed_schedule_moments': {}, 'recomputed_summaries': 0}
    with np.load(archive, allow_pickle=False) as arrays:
        for label, data in result['models'].items():
            model, _, _ = critical_model(label, cfg)
            calibration = data['calibration']
            scores = arrays[f'{label}__calibration_maxima']
            np.testing.assert_equal(np.sort(scores)[calibration['rank'] - 1], calibration['threshold'])
            np.testing.assert_equal(arrays[f'{label}__healthy_any_alarm'].mean(), calibration['healthy_path_FWER'])
            audit['fixed_schedule_moments'][label] = {}
            for fixture, record in data['fixtures'].items():
                prefix = f'{label}__{fixture}'
                active = np.array(record['fixture_active_mask'])
                for policy, metrics in record['policies'].items():
                    gate = arrays[f'{prefix}__{policy}__critical_gate'][:, 1:]
                    calculate_run = longest_run(gate)
                    np.testing.assert_array_equal(calculate_run, arrays[f'{prefix}__{policy}__max_rejection_run'])
                    expected_contract = calculate_run <= 6 if label == 'unstable' else np.ones(len(gate), bool)
                    np.testing.assert_array_equal(expected_contract,
                                                  arrays[f'{prefix}__{policy}__all_node26_seven_windows_contract'])
                    for metric, stats in metrics.items():
                        check_summary(arrays[f'{prefix}__{policy}__{metric}'], stats)
                        audit['recomputed_summaries'] += 1
                for pair, metrics in record['paired_differences'].items():
                    left, right = pair.split('_minus_')
                    for metric, stats in metrics.items():
                        diff = arrays[f'{prefix}__{left}__{metric}'] - arrays[f'{prefix}__{right}__{metric}']
                        check_summary(diff, stats)
                        audit['recomputed_summaries'] += 1
                cutoff_moments = timing_oracle_covariance(model, active)
                exact_mean = float(cutoff_moments[START - 1:STOP - 1].mean())
                observed = record['policies']['known_fault_schedule']['missing_window_mse']
                standard_errors = (observed['mean'] - exact_mean) / observed['path_se']
                assert abs(standard_errors) < 4
                if fixture == 'continuous':
                    assert exact_mean >= data['geometry']['blackout_run60_mean_lower_bound']
                audit['fixed_schedule_moments'][label][fixture] = {
                    'exact_window_expected_mse': exact_mean, 'simulated_mean': observed['mean'],
                    'simulation_minus_exact_in_standard_errors': standard_errors,
                }
        audit['path_arrays'] = len(arrays.files)
    print(json.dumps(audit, indent=2))
    (HERE / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
