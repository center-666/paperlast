"""Evaluate a critical complementary link under fixed offline sensor faults."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import block_diag, solve_discrete_are
from scipy.stats import beta as beta_dist


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARTIAL = ROOT / 'paper1_partial_observability_20261123'
PHYSICAL = ROOT / 'paper1_common_g50_gain_acceptance_20261121/physical_configuration.json'
sys.path.insert(0, str(PARTIAL))
from run_partial50 import design, matvec, obs_rank, physical, scores, summary
from watson_crn import beta_from_common, make_common_draws


RECEIVER = 26
SENDER = 35
EDGE = 6
T = 150
START, STOP = 50, 110


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def critical_model(label, cfg):
    model, previous = design(label, cfg)
    neighbors = cfg['graph']['neighbors'][RECEIVER]
    complementary = [j for j in neighbors if model.types[j] != model.types[RECEIVER]]
    assert neighbors == [16, 28, 35] and complementary == [SENDER]
    assert not np.any(model.links[:, 0] == RECEIVER)
    previous_edges = model.links.copy()
    existing = {(int(i), int(j)) for i, j in previous_edges}
    spare_counts = []
    for i, _ in previous_edges:
        spare_counts.append(int(sum(model.types[i] != model.types[j] and (int(i), j) not in existing
                                    for j in cfg['graph']['neighbors'][i])))
    assert spare_counts == [2, 8, 9, 2, 7, 2]

    remaining = [j for j in neighbors if j != SENDER]
    C = np.vstack([model.H[RECEIVER], model.H[remaining].sum(axis=0)])
    V = block_diag(model.R[RECEIVER], model.R[remaining].sum(axis=0))
    A2, Q2 = model.A2, model.Q[:2, :2]
    C2 = C[:, :2]
    prior = solve_discrete_are(A2.T, C2.T, Q2, V)
    gain = np.linalg.solve(C2 @ prior @ C2.T + V, C2 @ prior).T
    extended_gain = np.vstack([gain, np.zeros((2, 4))])
    drop_F = (np.eye(4) - extended_gain @ C) @ model.A
    keep_F = model.F[0, RECEIVER].copy()
    np.testing.assert_array_equal(drop_F[2:, 2:], A2)
    np.testing.assert_array_equal(drop_F[:2, 2:], 0)
    np.testing.assert_array_equal(drop_F[2:, :2], 0)
    assert obs_rank(model.A, C) == 2
    model.K = np.concatenate([model.K, model.K])
    model.G = np.concatenate([model.G, model.G])
    model.F = np.concatenate([model.F, model.F])
    model.K[64:, RECEIVER] = extended_gain[:, :2]
    model.G[64:, RECEIVER] = extended_gain[:, 2:]
    model.F[64:, RECEIVER] = drop_F

    parity, reference_covariance = model.source_functions['build_parity_space'](A2)
    model.Z = np.concatenate([model.Z, parity[SENDER][None]])
    model.Theta = np.concatenate([model.Theta, reference_covariance[SENDER][None]])
    model.links = np.vstack([previous_edges, [RECEIVER, SENDER]])
    observability_stack = np.vstack([model.H[SENDER] @ np.linalg.matrix_power(model.A, j) for j in range(5)])
    parity_error = np.linalg.norm(model.Z[EDGE] @ observability_stack)
    assert parity_error < 1e-10

    noise = np.zeros((2, 2))
    blackout = []
    for _ in range(STOP - START):
        noise = A2 @ noise @ A2.T + Q2
        blackout.append(float(np.trace(noise)))

    other_norm = previous['max_norm_F_2']
    drop_norm = np.linalg.norm(drop_F, 2)
    keep_norm = np.linalg.norm(keep_F, 2)
    global_five_bound = max(drop_norm, other_norm)**4 * other_norm
    node_seven_bound = max(max(drop_norm, keep_norm)**6 * keep_norm, other_norm**7)
    words = list(itertools.product([False, True], repeat=7))
    window_norms = np.empty(2**7)
    for code, word in enumerate(words):
        product = np.eye(4)
        for rejected in word:
            product = (drop_F if rejected else keep_F) @ product
        window_norms[code] = np.linalg.norm(product, 2)
    assert node_seven_bound < 1 and np.max(window_norms[:-1]) <= node_seven_bound + 1e-12
    geometry = {
        'neighbors': neighbors, 'only_complementary_neighbor': SENDER,
        'original_edges_remaining_complement_counts': spare_counts,
        'original_edge_is_complementary': [bool(model.types[i] != model.types[j]) for i, j in previous_edges],
        'rejected_mode_observability_rank': obs_rank(model.A, C),
        'missing_block_open_loop_radius': float(max(abs(np.linalg.eigvals(A2)))),
        'parity_annihilation_residual': float(parity_error),
        'keep_K': model.K[0, RECEIVER].tolist(), 'keep_G': model.G[0, RECEIVER].tolist(),
        'drop_K': model.K[64, RECEIVER].tolist(), 'drop_G': model.G[64, RECEIVER].tolist(),
        'keep_F': keep_F.tolist(), 'drop_F': drop_F.tolist(),
        'keep_norm': float(keep_norm), 'drop_norm': float(drop_norm),
        'other_receiver_norm_upper_bound': other_norm,
        'conservative_global_five_window_upper_bound': float(global_five_bound),
        'conservative_node_specific_seven_window_upper_bound': float(node_seven_bound),
        'node26_enumerated_max_seven_window_norm_with_keep': float(np.max(window_norms[:-1])),
        'node26_all_drop_seven_window_norm': float(window_norms[-1]),
        'exogenous_blackout_lower_bound_trace': blackout,
        'blackout_run60_mean_lower_bound': float(np.mean(blackout)),
    }
    return model, geometry, window_norms


def innovation_aggregate(model, paths):
    return np.einsum('ij,btjm->btim', model.adjacency, paths.y, optimize=True)


def run_filter(model, paths, aggregate, delta, gates, literal=False):
    count = len(paths.x)
    masks = gates @ (1 << np.arange(7))
    post = np.zeros((count, T, 50, 4))
    forcing = np.zeros((count, T))
    edges = {tuple(edge): i for i, edge in enumerate(model.links)}
    for k in range(1, T):
        p = post[:, k - 1] @ model.A.T
        K = model.K[masks[:, k]]
        G = model.G[masks[:, k]]
        current = paths.y[:, k]
        local = current - matvec(model.H, p)
        if literal:
            neighbor_innovation = np.zeros((count, 50, 2))
            for i, neighbors in enumerate(model.cfg['graph']['neighbors']):
                for j in neighbors:
                    pair = edges.get((i, j))
                    report = current[:, j] - matvec(model.H[j], p[:, i])
                    if pair is not None:
                        report = (~gates[:, k, pair, None]) * (report + delta[:, k, pair])
                    neighbor_innovation[:, i] += report
            post[:, k] = p + matvec(K, local) + matvec(G, neighbor_innovation)
        else:
            received = aggregate[:, k].copy()
            for edge, (i, j) in enumerate(model.links):
                drop = gates[:, k, edge, None]
                received[:, i] += (~drop) * delta[:, k, edge] - drop * current[:, j]
            post[:, k] = matvec(model.F[masks[:, k]], post[:, k - 1]) + matvec(K, current) + matvec(G, received)
        admitted = (~gates[:, k, EDGE, None]) * delta[:, k, EDGE]
        forcing[:, k] = np.sum(matvec(G[:, RECEIVER, 2:, :], admitted)**2, axis=-1)
    return post, forcing


def metrics(post, forcing, paths, gates, active, window_norms):
    squared_error = (post - paths.x[:, :, None, :])**2
    missing = squared_error[:, :, RECEIVER, 2:].sum(axis=-1)
    reject = gates[:, :, EDGE]
    scoring_window = reject[:, START:STOP]
    rolling = np.lib.stride_tricks.sliding_window_view(reject[:, 1:], 7, axis=1)
    codes = rolling @ (1 << np.arange(6, -1, -1))
    run = np.zeros(len(post), dtype=int)
    max_rejection_run = np.zeros_like(run)
    for column in reject[:, 1:].T:
        run = (run + 1) * column
        max_rejection_run = np.maximum(max_rejection_run, run)
    out = {
        'missing_window_mse': missing[:, START:STOP].mean(axis=1),
        'missing_post20_mse': missing[:, 20:].mean(axis=1),
        'missing_end_mse': missing[:, STOP - 1],
        'missing_tail_mse': missing[:, STOP:].mean(axis=1),
        'network_window_mse': squared_error[:, START:STOP].sum(axis=-1).mean(axis=(1, 2)),
        'network_post20_mse': squared_error[:, 20:].sum(axis=-1).mean(axis=(1, 2)),
        'max_rejection_run': max_rejection_run,
        'max_seven_window_node26_norm': np.max(window_norms[codes], axis=1),
        'all_node26_seven_windows_contract': np.max(window_norms[codes], axis=1) < 1,
        'critical_window_reject_fraction': scoring_window.mean(axis=1),
        'accepted_clean_fraction': (~reject[:, 1:][:, ~active[1:]]).mean(axis=1),
        'admitted_missing_block_forcing_window': forcing[:, START:STOP].mean(axis=1),
    }
    if np.any(active):
        out['accepted_perturbed_fraction'] = (~reject[:, active]).mean(axis=1)
    if np.any(~active[START:STOP]):
        out['accepted_clean_inside_window_fraction'] = (~scoring_window[:, ~active[START:STOP]]).mean(axis=1)
    return out


def evaluate_fixture(model, reg, fixture, seed, threshold, norms, arrays, prefix):
    count = reg['paths_per_fixture_and_model']
    paths = physical(model, count, seed)
    aggregate = innovation_aggregate(model, paths)
    draws = make_common_draws(count, STOP - START, 1, seed + 1, seed + 2, seed + 3)
    indices = np.arange(T)
    active = (indices >= START) & (indices < STOP)
    if fixture == 'burst':
        active &= (indices - START) % 10 < 6
    error = np.asarray(reg['fixture_vector']) * active[START:STOP, None]
    zero = np.zeros((count, T, 7, 2))
    clean_gates = np.zeros((count, T, 7), bool)
    timing_gates = clean_gates.copy()
    timing_gates[:, :, EDGE] = active
    permanent_gates = clean_gates.copy()
    permanent_gates[:, 1:, EDGE] = True
    policy_metrics = {}

    def evaluate(name, delta, gates):
        post, forcing = run_filter(model, paths, aggregate, delta, gates)
        result = metrics(post, forcing, paths, gates, active, norms)
        policy_metrics[name] = result
        for metric, values in result.items():
            arrays[f'{prefix}__{name}__{metric}'] = values
        arrays[f'{prefix}__{name}__critical_gate'] = gates[:, :, EDGE]

    for name, theta in reg['theta_candidates'].items():
        beta = beta_from_common(draws, theta)[:, :, 0, :]
        delta = zero.copy()
        delta[:, START:STOP, EDGE] = .5 * error + beta * np.sum(beta * error, axis=-1, keepdims=True)
        detector_gates = scores(model, paths, delta) > threshold
        evaluate(name, delta, detector_gates)
        evaluate(f'{name}_always_accept', delta, clean_gates)
        if name == 'iid':
            from types import SimpleNamespace

            small = SimpleNamespace(x=paths.x[:2], y=paths.y[:2])
            masks = np.arange(T) % 128
            forced = np.broadcast_to((masks[:, None] & (1 << np.arange(7))) > 0, (2, T, 7))
            arguments = (model, small, aggregate[:2], delta[:2], forced)
            recursive, _ = run_filter(*arguments)
            literal, _ = run_filter(*arguments, literal=True)
            np.testing.assert_allclose(recursive, literal, rtol=1e-10, atol=5e-9)
            literal_error = float(np.max(np.abs(recursive - literal)))

    evaluate('permanent_reject', zero, permanent_gates)
    evaluate('known_fault_schedule', zero, timing_gates)
    evaluate('clean_reference', zero, clean_gates)
    comparisons = {}
    for left, right in [('negative', 'iid'), ('positive', 'iid'), ('negative', 'negative_always_accept'),
                        ('negative', 'known_fault_schedule'), ('negative', 'permanent_reject')]:
        differences = {}
        for metric in ('missing_window_mse', 'missing_post20_mse', 'missing_tail_mse'):
            differences[metric] = summary(policy_metrics[left][metric] - policy_metrics[right][metric])
        comparisons[f'{left}_minus_{right}'] = differences
    print(prefix, 'node26 window MSE', {k: round(float(v['missing_window_mse'].mean()), 4)
          for k, v in policy_metrics.items()}, flush=True)
    return {'physical_seed': seed, 'watermark_seeds': [seed + 1, seed + 2, seed + 3],
            'fixture_active_mask': active.tolist(), 'literal_max_error': literal_error,
            'policies': {name: {metric: summary(v.astype(float)) for metric, v in m.items()}
                         for name, m in policy_metrics.items()}, 'paired_differences': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    registration_path = HERE / 'registration.json'
    reg = json.loads(registration_path.read_text(encoding='utf-8'))
    cfg = json.loads(PHYSICAL.read_text(encoding='utf-8'))
    sources = {
        PHYSICAL: 'c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d',
        PARTIAL / 'run_partial50.py': '94b53739598ea46b2dc40595b338a6d368d3d5e398d1e2797e785582e17b026e',
        ROOT / 'paper1_sscd_commonrandom_20261122/watson_crn.py': 'e3c3af17fed3a6afb2c71aa8a33ede942ba80796b7b34862ec6768bc59068f7a',
    }
    for source, expected in sources.items():
        if digest(source) != expected:
            raise ValueError(f'Source hash mismatch: {source}')
    result_path = HERE / 'critical_results.json'
    archive_path = HERE / 'critical_paths.npz'
    if args.execute and (result_path.exists() or archive_path.exists()):
        raise FileExistsError('Existing run results. Use a fresh checkout for replication.')
    result = {'script_sha256': digest(Path(__file__)), 'registration_sha256': digest(registration_path),
              'sources': {p.relative_to(ROOT).as_posix(): expected for p, expected in sources.items()},
              'models': {}}
    arrays = {}
    for model_index, label in enumerate(('unstable', 'stable')):
        model, geometry, norms = critical_model(label, cfg)
        result['models'][label] = {'geometry': geometry}
        if not args.execute:
            print(label, 'drop rho', geometry['missing_block_open_loop_radius'],
                  '60-step bound', geometry['exogenous_blackout_lower_bound_trace'][-1],
                  '7-window bound', geometry['conservative_node_specific_seven_window_upper_bound'])
            continue
        base_seed = reg['root_seed'] + model_index * 10000
        ncal = reg['paths_per_calibration_model']
        nhealthy = reg['paths_per_healthy_holdout_model']
        cal = physical(model, ncal, base_seed + 1)
        maxima = scores(model, cal, np.zeros((ncal, T, 7, 2)))[:, 4:].max(axis=(1, 2))
        rank = int(np.ceil((len(maxima) + 1) * .95))
        threshold = float(np.partition(maxima, rank - 1)[rank - 1])
        healthy = physical(model, nhealthy, base_seed + 2)
        healthy_gates = scores(model, healthy, np.zeros((nhealthy, T, 7, 2))) > threshold
        hits = healthy_gates[:, 4:].any(axis=(1, 2))
        hit_count = int(hits.sum())
        arrays[f'{label}__calibration_maxima'] = maxima
        arrays[f'{label}__healthy_critical_gate'] = healthy_gates[:, :, EDGE]
        arrays[f'{label}__healthy_any_alarm'] = hits
        result['models'][label]['calibration'] = {
            'rank': rank, 'threshold': threshold, 'seed': base_seed + 1, 'healthy_holdout_seed': base_seed + 2,
            'healthy_path_false_alarms': hit_count, 'healthy_holdout_paths': nhealthy,
            'healthy_path_FWER': float(hits.mean()), 'healthy_critical_path_alarm': float(healthy_gates[:, :, EDGE].any(axis=1).mean()),
            'FWER_exact_binomial_CI95': [float(beta_dist.ppf(.025, hit_count, nhealthy - hit_count + 1)) if hit_count else 0.,
                                        float(beta_dist.ppf(.975, hit_count + 1, nhealthy - hit_count)) if hit_count < nhealthy else 1.],
        }
        result['models'][label]['fixtures'] = {}
        for index, fixture in enumerate(reg['fixtures']):
            prefix = f'{label}__{fixture}'
            result['models'][label]['fixtures'][fixture] = evaluate_fixture(
                model, reg, fixture, base_seed + 100 + index * 10, threshold, norms, arrays, prefix)
    if args.execute:
        with TemporaryDirectory(dir=HERE) as temporary:
            draft_archive = Path(temporary) / archive_path.name
            draft_result = Path(temporary) / result_path.name
            np.savez_compressed(draft_archive, **arrays)
            result['path_archive_sha256'] = digest(draft_archive)
            save_json(draft_result, result)
            draft_archive.rename(archive_path)
            draft_result.rename(result_path)
        print(f'Completed offline evaluation and saved {len(arrays)} arrays.')


if __name__ == '__main__':
    main()
