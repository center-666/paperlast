"""Read-back of saved paired arrays and published JSON; no new simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def close(actual, expected):
    if not np.allclose(actual, expected, rtol=1e-10, atol=2e-12):
        raise AssertionError((actual, expected))


def main():
    counts = {'status': 'PASS', 'new_physical_paths': 0, 'models': {}}
    for label in ['unstable', 'stable']:
        report = json.loads((HERE / f'{label}_validation.json').read_text(encoding='utf-8'))
        assert report['paths'] == 1024 and 'stacked KF' in report['estimator']
        arrays = np.load(HERE / f'{label}_paired_paths.npz', allow_pickle=False)
        for key in arrays.files:
            value = arrays[key]
            assert value.shape[0] == 1024 and np.all(np.isfinite(value)), key
        for key, row in report['summaries'].items():
            value = arrays[key].astype(float)
            assert value.shape == (1024,)
            close(value.mean(), row['mean'])
            close(value.std(ddof=1) / np.sqrt(1024), row['path_se'])
        for key, row in report['paired_comparisons'].items():
            lhs, metric = key.split('__')
            lhs = lhs.removesuffix('_minus_iid')
            mode, _ = lhs.rsplit('_', 1)
            difference = arrays[lhs + '__' + metric] - arrays[mode + '_iid__' + metric]
            close(difference.mean(), row['mean'])
            close(difference.std(ddof=1) / np.sqrt(1024), row['path_se'])
        for key, row in report['delay_summaries'].items():
            value = arrays[key]
            detected = value < 150
            close(detected.mean(axis=0), row['per_link_detected_fraction'])
            for e, expected in enumerate(row['per_link_conditional_mean']):
                if expected is None:
                    assert not detected[:, e].any()
                else:
                    close(value[:, e][detected[:, e]].mean(), expected)
        for mode in ['wm_no_gate', 'legacy42', 'calibrated']:
            for kernel in ['iid', 'negative', 'positive']:
                key = mode + '_' + kernel + '__'
                close(arrays[key + 'paired_mse_increment_network_post20'],
                      arrays[key + 'displacement_mse_network_post20']
                      + arrays[key + 'state_cross_term_network_post20'])
        cal = report['calibration']
        assert len(cal['path_maxima']) == cal['paths'] == 512
        close(np.quantile(cal['path_maxima'], .95, method='higher'), cal['calibrated_pathmax_threshold'])
        assert report['SSCD_selector_alias'] == 'negative'
        assert report['identity_checks']['post117_gate_disagreement'] == 0
        counts['models'][label] = {'paths': 1024, 'calibration_paths': 512,
                                  'saved_arrays': len(arrays.files),
                                  'scalar_summaries_verified': len(report['summaries']),
                                  'paired_comparisons_verified': len(report['paired_comparisons']),
                                  'delay_tables_verified': len(report['delay_summaries'])}
    source = HERE.parent / 'paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha == '2f4694e6180680ccd83ecc4ae60cb587b76276625c9d34c5946cfeca2d3e8a4a'
    counts['source_sha256'] = sha
    (HERE / 'archive_checks.json').write_text(json.dumps(counts, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
