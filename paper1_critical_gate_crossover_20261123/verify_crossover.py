"""Read back the full seven-edge gate traces and paired loss decomposition."""

import numpy as np

from run_crossover import ARCHIVED_METRICS, BASE, HERE, ROOT, T, digest, read, summary


def verify_summary(values, expected, count):
    assert values.shape == (count,) and np.isfinite(values).all()
    actual = summary(values)
    for statistic in actual:
        np.testing.assert_allclose(actual[statistic], expected[statistic], atol=1e-12, rtol=1e-12)


def main():
    report = read(HERE / 'crossover.json')
    archive = HERE / 'crossover.npz'
    assert report['script_sha256'] == digest(HERE / 'run_crossover.py')
    assert report['path_archive_sha256'] == digest(archive)
    for path, source in report['sources'].items():
        assert digest(ROOT / path) == source['sha256'], path

    summaries = 0
    decompositions = 0
    gate_bits = 0
    historical_path_values = 0
    with np.load(archive, allow_pickle=False) as paths, np.load(BASE / 'critical_paths.npz', allow_pickle=False) as old:
        assert len(paths.files) == report['path_array_count']
        for label, fixtures in report['models'].items():
            for fixture, record in fixtures.items():
                prefix = f'{label}__{fixture}'
                count = record['paths']
                for policy, metrics in record['policies'].items():
                    key = f'{prefix}__{policy}'
                    gates = paths[f'{key}__frozen_gates']
                    assert gates.shape == (count, T, 7) and gates.dtype == np.bool_
                    np.testing.assert_array_equal(gates[:, :, 6], old[f'{key}__critical_gate'])
                    gate_bits += count * T

                    for metric, statistics in metrics.items():
                        values = {term: paths[f'{key}__{metric}__{term}'] for term in statistics}
                        for term, expected in statistics.items():
                            verify_summary(values[term], expected, count)
                            summaries += 1
                        a, b, c, d = (values[name] for name in ('JG_fault', 'JG_clean', 'J0_fault', 'J0_clean'))
                        calculations = {
                            'selection_cost': b - d,
                            'residual_damage': a - b,
                            'damage_without_gate': c - d,
                            'damage_removed_by_gate': (c - d) - (a - b),
                            'net_gate_cost': a - c,
                        }
                        for term, expected in calculations.items():
                            np.testing.assert_allclose(values[term], expected, atol=1e-12, rtol=1e-12)
                        np.testing.assert_allclose(a - c, (b - d) + ((a - b) - (c - d)), atol=1e-10, rtol=1e-12)
                        decompositions += count
                        if metric.startswith('missing_'):
                            reconstructed = values['residual_displacement_energy'] + values['residual_cross_term']
                            np.testing.assert_allclose(a - b, reconstructed, atol=5e-8, rtol=1e-9)
                        if metric in ARCHIVED_METRICS:
                            suffix = ARCHIVED_METRICS[metric]
                            for branch, previous in ((a, policy), (c, f'{policy}_always_accept'), (d, 'clean_reference')):
                                np.testing.assert_allclose(branch, old[f'{prefix}__{previous}__{suffix}'],
                                                           atol=1e-8, rtol=1e-10)
                                historical_path_values += count

                for metric, statistics in record['negative_minus_iid'].items():
                    for term, expected in statistics.items():
                        negative = paths[f'{prefix}__negative__{metric}__{term}']
                        iid = paths[f'{prefix}__iid__{metric}__{term}']
                        verify_summary(negative - iid, expected, count)
                        summaries += 1

    print(f'PASS: {report["path_array_count"]} arrays, {summaries} statistical summaries, '
          f'{decompositions} path-window decompositions, {gate_bits} historical gate bits, '
          f'{historical_path_values} historical path-loss values.')


if __name__ == '__main__':
    main()
