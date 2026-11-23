"""Replay the four fixed worlds for the archived 35-to-26 offline fault fixtures."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'paper1_critical_complement_20261123'
COMMIT = '78160f3de4e210aa205752bfefc5ce5b0780b7f9'
sys.path.insert(0, str(BASE))
from run_critical import (EDGE, PHYSICAL, RECEIVER, START, STOP, T, beta_from_common,
                          critical_model, digest, innovation_aggregate, make_common_draws,
                          matvec, physical, run_filter, save_json, scores, summary)


WINDOWS = {'fault_window': slice(START, STOP), 'after_fault': slice(STOP, T), 'post20': slice(20, T)}
ARCHIVED_METRICS = {
    'missing_fault_window': 'missing_window_mse', 'missing_after_fault': 'missing_tail_mse',
    'missing_post20': 'missing_post20_mse', 'network_fault_window': 'network_window_mse',
    'network_post20': 'network_post20_mse',
}
WORLDS = ('JG_fault', 'JG_clean', 'J0_fault', 'J0_clean')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def git_blob(path):
    return subprocess.check_output(['git', 'rev-parse', f'{COMMIT}:{path.relative_to(ROOT).as_posix()}'],
                                   cwd=ROOT, text=True).strip()


def bind_sources(record):
    paths = [BASE / name for name in ('run_critical.py', 'registration.json', 'critical_results.json',
                                      'critical_paths.npz')]
    paths.extend(ROOT / name for name in record['sources'])
    paths.extend(ROOT / name for name in (
        'paper1_50node_tv_gate_theory_20261121/replay.py',
        'paper1_50node_tv_gate_theory_20261121/source_model.py',
        'paper1_original50_source_20261121/configuration_snapshot.json',
        'paper1_original50_source_20261121/source/simulation_revised_detection_metrics_complete_v2.py',
    ))
    manifest = {}
    for path in paths:
        expected = git_blob(path)
        working = subprocess.check_output(['git', 'hash-object', '--no-filters', str(path)],
                                          cwd=ROOT, text=True).strip()
        assert expected == working, path
        manifest[path.relative_to(ROOT).as_posix()] = {'git_blob': expected, 'sha256': digest(path)}
    return manifest


def reduce_losses(series):
    return {f'{scope}_{window}': values[:, period].mean(axis=1)
            for scope, values in series.items() for window, period in WINDOWS.items()}


def loss_series(post, paths):
    squared = (post - paths.x[:, :, None, :])**2
    return {'missing': squared[:, :, RECEIVER, 2:].sum(axis=-1),
            'network': squared.sum(axis=-1).mean(axis=-1)}


def pointwise_displacement(model, gates, delta):
    """Propagate the full node-local difference before extracting the missing block."""
    assert np.flatnonzero(model.links[:, 0] == RECEIVER).tolist() == [EDGE]
    masks = gates[:, :, EDGE].astype(np.intp) * 64
    response = np.zeros((len(gates), T, model.A.shape[0]))
    for tick in range(1, T):
        F = model.F[masks[:, tick], RECEIVER]
        G = model.G[masks[:, tick], RECEIVER]
        accepted = (~gates[:, tick, EDGE, None]) * delta[:, tick, EDGE]
        response[:, tick] = matvec(F, response[:, tick - 1]) + matvec(G, accepted)
    return response[:, :, 2:]


def check_old_losses(losses, archive, prefix, policy):
    maximum = 0.
    for key, old_key in ARCHIVED_METRICS.items():
        old = archive[f'{prefix}__{policy}__{old_key}']
        np.testing.assert_allclose(losses[key], old, atol=1e-8, rtol=1e-10)
        maximum = max(maximum, float(np.max(np.abs(losses[key] - old))))
    return maximum


def run_fixture(label, fixture, model, record, threshold, registration, old, out):
    count = registration['paths_per_fixture_and_model']
    prefix = f'{label}__{fixture}'
    paths = physical(model, count, record['physical_seed'])
    aggregate = innovation_aggregate(model, paths)
    draws = make_common_draws(count, STOP - START, 1, *record['watermark_seeds'])
    active = np.asarray(record['fixture_active_mask'], dtype=bool)
    fixture_error = np.asarray(registration['fixture_vector']) * active[START:STOP, None]
    zero = np.zeros((count, T, 7, 2))
    keep = np.zeros((count, T, 7), bool)
    reference, _ = run_filter(model, paths, aggregate, zero, keep)
    clean_losses = reduce_losses(loss_series(reference, paths))
    archive_error = check_old_losses(clean_losses, old, prefix, 'clean_reference')
    result = {'physical_seed': record['physical_seed'], 'watermark_seeds': record['watermark_seeds'],
              'threshold': threshold, 'paths': count, 'policies': {}, 'negative_minus_iid': {}}
    displacement_error = 0.
    path_metrics = {}
    for policy, theta in registration['theta_candidates'].items():
        beta = beta_from_common(draws, theta)[:, :, 0]
        delta = zero.copy()
        delta[:, START:STOP, EDGE] = .5 * fixture_error + beta * np.sum(beta * fixture_error, axis=-1, keepdims=True)
        frozen_gates = scores(model, paths, delta) > threshold
        np.testing.assert_array_equal(frozen_gates[:, :, EDGE], old[f'{prefix}__{policy}__critical_gate'])
        post_A_G, _ = run_filter(model, paths, aggregate, delta, frozen_gates)
        post_0_G, _ = run_filter(model, paths, aggregate, zero, frozen_gates)
        post_A_0, _ = run_filter(model, paths, aggregate, delta, keep)
        worlds = dict(zip(WORLDS, (post_A_G, post_0_G, post_A_0, reference)))
        losses = {key: reduce_losses(loss_series(value, paths)) for key, value in worlds.items()}
        archive_error = max(archive_error, check_old_losses(losses['JG_fault'], old, prefix, policy),
                            check_old_losses(losses['J0_fault'], old, prefix, f'{policy}_always_accept'))

        # Separate the quadratic displacement from its interaction with the clean replay error.
        b = pointwise_displacement(model, frozen_gates, delta)
        d = post_A_G[:, :, RECEIVER, 2:] - post_0_G[:, :, RECEIVER, 2:]
        np.testing.assert_allclose(d, b, atol=2e-8, rtol=1e-10)
        displacement_error = max(displacement_error, float(np.max(np.abs(d-b))))
        clean_error = post_0_G[:, :, RECEIVER, 2:] - paths.x[:, :, 2:]
        energy = np.sum(b*b, axis=-1)
        cross = 2 * np.sum(clean_error*b, axis=-1)
        for interval in WINDOWS.values():
            np.testing.assert_allclose((np.sum(d*d, axis=-1) + 2*np.sum(clean_error*d, axis=-1))[:, interval].mean(axis=1),
                                       (energy + cross)[:, interval].mean(axis=1), atol=5e-8, rtol=1e-9)

        out[f'{prefix}__{policy}__frozen_gates'] = frozen_gates
        result['policies'][policy] = {}
        path_metrics[policy] = {}
        for key in clean_losses:
            terms = {world: losses[world][key] for world in WORLDS}
            terms['selection_cost'] = terms['JG_clean'] - terms['J0_clean']
            terms['residual_damage'] = terms['JG_fault'] - terms['JG_clean']
            terms['damage_without_gate'] = terms['J0_fault'] - terms['J0_clean']
            terms['damage_removed_by_gate'] = terms['damage_without_gate'] - terms['residual_damage']
            terms['net_gate_cost'] = terms['JG_fault'] - terms['J0_fault']
            np.testing.assert_allclose(terms['net_gate_cost'], terms['selection_cost'] - terms['damage_removed_by_gate'],
                                       atol=1e-10, rtol=1e-12)
            if key.startswith('missing_'):
                period = WINDOWS[key.removeprefix('missing_')]
                terms['residual_displacement_energy'] = energy[:, period].mean(axis=1)
                terms['residual_cross_term'] = cross[:, period].mean(axis=1)
                np.testing.assert_allclose(terms['residual_damage'],
                                           terms['residual_displacement_energy'] + terms['residual_cross_term'],
                                           atol=5e-8, rtol=1e-9)
            path_metrics[policy][key] = terms
            result['policies'][policy][key] = {name: summary(value) for name, value in terms.items()}
            for name, value in terms.items():
                out[f'{prefix}__{policy}__{key}__{name}'] = value
    for key in clean_losses:
        diffs = {name: path_metrics['negative'][key][name] - path_metrics['iid'][key][name]
                 for name in ('JG_fault', 'selection_cost', 'residual_damage')}
        np.testing.assert_allclose(diffs['JG_fault'], diffs['selection_cost'] + diffs['residual_damage'],
                                   atol=1e-10, rtol=1e-12)
        result['negative_minus_iid'][key] = {name: summary(value) for name, value in diffs.items()}
    result['max_historical_path_loss_difference'] = archive_error
    result['max_missing_block_displacement_residual'] = displacement_error
    print(prefix, {name: tuple(round(metrics['missing_fault_window'][key]['mean'], 4) for key in
                              ('selection_cost', 'residual_damage', 'net_gate_cost'))
                   for name, metrics in result['policies'].items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_json, output_npz = (output / f'crossover.{ext}' for ext in ('json', 'npz'))
    if output_json.exists() or output_npz.exists():
        raise FileExistsError('Use a new --output-dir to reproduce the archived result.')

    baseline = read(BASE / 'critical_results.json')
    sources = bind_sources(baseline)
    cfg = read(PHYSICAL)
    registration = read(BASE / 'registration.json')
    result = {'source_commit': COMMIT, 'sources': sources, 'script_sha256': digest(Path(__file__)),
              'study': 'Post hoc paired replay of all original 35-to-26 offline fault trajectories',
              'independent_performance_sample': False,
              'worlds': dict(zip(WORLDS, ('faulty with frozen gates', 'clean with frozen gates',
                                        'faulty all accepted', 'clean all accepted'))),
              'windows_half_open': {name: [p.start, p.stop] for name, p in WINDOWS.items()}, 'models': {}}
    arrays = {}
    with np.load(BASE / 'critical_paths.npz', allow_pickle=False) as old:
        for label, record in baseline['models'].items():
            model, _, _ = critical_model(label, cfg)
            result['models'][label] = {fixture: run_fixture(label, fixture, model, fixture_record,
                                      record['calibration']['threshold'], registration, old, arrays)
                                      for fixture, fixture_record in record['fixtures'].items()}
    with TemporaryDirectory(dir=output) as staging:
        npz = Path(staging) / output_npz.name
        report = Path(staging) / output_json.name
        np.savez_compressed(npz, **arrays)
        result['path_archive_sha256'] = digest(npz)
        result['path_array_count'] = len(arrays)
        save_json(report, result)
        npz.rename(output_npz)
        report.rename(output_json)


if __name__ == '__main__':
    main()
