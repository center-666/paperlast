"""Recompute nominal aggregate-observation common-G gains, not historical gains.

The only physical input is the frozen 049bdc6 configuration. The 57 rows per
dynamics are NODE-LOCAL modes; the mapping covers all 64 joint gate masks.
No original Monte Carlo wrapper or optimization routine is imported.
"""
from pathlib import Path
import hashlib
import itertools
import json
import platform
import numpy as np
import scipy
from scipy.linalg import block_diag, solve_discrete_are

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'paper1_original50_source_20261121/configuration_snapshot.json'
EXPECTED_CONFIGURATION_SHA256 = 'c25437077cb73ac2332160e55f6183550bb8fc5318d93bb32c183a11d661b50d'


def aggregate_matrices(H, R, receiver, active):
    """The neighbor block is an UNWEIGHTED sum; self is a separate block."""
    if receiver in active:
        raise ValueError('Self must not occur in the neighbor aggregate.')
    if not active:
        return H[receiver].copy(), R[receiver].copy()
    return np.vstack([H[receiver], H[active].sum(axis=0)]), block_diag(
        R[receiver], R[active].sum(axis=0))


def solve_mode(A, Q, H, R, receiver, active):
    C, V = aggregate_matrices(H, R, receiver, active)
    J = C.T @ np.linalg.solve(V, C)
    P = 5.0 * np.eye(2)
    for iteration in range(1, 1001):
        prior = A @ P @ A.T + Q
        next_P = np.linalg.inv(np.linalg.inv(prior) + J)
        difference = np.linalg.norm(next_P - P, 'fro')
        P = next_P
        if difference <= 1e-14:
            break
    else:
        raise ArithmeticError('Posterior fixed-point iteration did not converge.')
    prior = A @ P @ A.T + Q
    post = np.linalg.inv(np.linalg.inv(prior) + J)
    L = np.linalg.solve(C @ prior @ C.T + V, C @ prior).T
    B = np.eye(2) - L @ C
    F = B @ A
    joseph = B @ prior @ B.T + L @ V @ L.T
    scipy_prior = solve_discrete_are(A.T, C.T, Q, V)
    K = L[:, :2]
    G = L[:, 2:] if active else np.zeros((2, 2))
    # Embedded gain acts on all physical sensor coordinates, not just two blocks.
    Lbar = np.zeros((2, 100))
    Lbar[:, 2 * receiver:2 * receiver + 2] = K
    for sender in active:
        Lbar[:, 2 * sender:2 * sender + 2] = G
    return {
        'receiver': int(receiver), 'active_neighbors': [int(j) for j in active],
        'retained_neighbor_count': len(active),
        'C': C.tolist(), 'V': V.tolist(), 'J': J.tolist(),
        'P_prior': prior.tolist(), 'P_posterior_nominal': post.tolist(),
        'K': K.tolist(), 'G': G.tolist(), 'F_diagnostic': F.tolist(),
        'fixed_point_iterations': iteration,
        'last_posterior_step_fro': float(difference),
        'posterior_fixed_point_residual_fro': float(np.linalg.norm(post - P, 'fro')),
        'prediction_dare_residual_fro': float(np.linalg.norm(A @ post @ A.T + Q - prior, 'fro')),
        'joseph_residual_fro': float(np.linalg.norm(joseph - post, 'fro')),
        'independent_scipy_prior_difference_fro': float(np.linalg.norm(scipy_prior - prior, 'fro')),
        'rho_F': float(max(abs(np.linalg.eigvals(F)))),
        'norm_F_2': float(np.linalg.norm(F, 2)),
        'norm_K_fro': float(np.linalg.norm(K, 'fro')),
        'norm_G_fro': float(np.linalg.norm(G, 'fro')),
        'norm_Lbar_2': float(np.linalg.norm(Lbar, 2)),
        'embedded_gain_bound': float(1 / np.sqrt(1 + len(active))),
        'minimum_prior_eigenvalue': float(np.linalg.eigvalsh(prior).min()),
        'minimum_posterior_eigenvalue': float(np.linalg.eigvalsh(post).min()),
    }


def summarize(rows):
    result = {'local_mode_count': len(rows)}
    for field in ['fixed_point_iterations', 'prediction_dare_residual_fro',
                  'posterior_fixed_point_residual_fro', 'joseph_residual_fro',
                  'independent_scipy_prior_difference_fro', 'rho_F', 'norm_F_2']:
        selected = max(rows, key=lambda x: x[field])
        result['max_' + field] = selected[field]
        if field in ['rho_F', 'norm_F_2']:
            result[field + '_maximizer'] = {
                'receiver': selected['receiver'],
                'dropped_link_indices': selected['dropped_link_indices']}
    for field in ['norm_K_fro', 'norm_G_fro', 'minimum_prior_eigenvalue',
                  'minimum_posterior_eigenvalue']:
        result[field + '_range'] = [min(x[field] for x in rows), max(x[field] for x in rows)]
    return result


def main():
    cfg_bytes = FROZEN.read_bytes()
    if hashlib.sha256(cfg_bytes).hexdigest() != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError('Input bytes are not the pinned 049bdc6 physical configuration.')
    cfg = json.loads(cfg_bytes)
    (HERE / 'physical_configuration.json').write_bytes(cfg_bytes)
    H, R, Q = (np.asarray(cfg[k], float) for k in ['H', 'R', 'Q'])
    links = [(e['receiver'], e['sender']) for e in cfg['attack_links']]
    neighbors = cfg['graph']['neighbors']
    table = {
        'status': 'NEW nominal aggregate-MMSE gain design; NOT recovered historical common-G gains',
        'input_baseline_commit': '049bdc6f657c6746678df5138c9b0c02f29dedbc',
        'physical_configuration_sha256': hashlib.sha256(cfg_bytes).hexdigest(),
        'attack_links_receiver_sender': links,
        'mask_convention': 'bit e=1 drops link e; gates are current-time, non-latched',
        'gain_policy': 'Precomputed steady nominal DARE gain for CURRENT local active set; no recursive covariance memory',
        'nominal_covariance_warning': 'DARE P is not the true conditional or switched estimation covariance under data-dependent gates',
        'iteration_tolerance_fro': 1e-14,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'models': {},
    }
    for label in ['unstable', 'stable']:
        A = np.asarray(cfg['A_' + label], float)
        rows, local_lookup = [], {}
        for receiver in range(50):
            incident = [e for e, (i, _) in enumerate(links) if i == receiver]
            for choices in itertools.product([False, True], repeat=len(incident)):
                dropped = [e for e, choice in zip(incident, choices) if choice]
                removed_senders = [links[e][1] for e in dropped]
                active = [int(j) for j in neighbors[receiver] if j not in removed_senders]
                row = solve_mode(A, Q, H, R, receiver, active)
                row['dropped_link_indices'] = dropped
                row['dropped_senders'] = removed_senders
                row['row_index'] = len(rows)
                local_lookup[(receiver, tuple(dropped))] = len(rows)
                rows.append(row)
        mapping = [[local_lookup[(i, tuple(e for e, (r, _) in enumerate(links)
                                         if r == i and (mask & (1 << e))))]
                    for i in range(50)] for mask in range(64)]
        baseline_rows = [rows[index] for index in mapping[0]]
        table['models'][label] = {
            'A': A.tolist(), 'baseline_summary': summarize(baseline_rows),
            'registered_modes_summary': summarize(rows),
            'global_mask_to_local_row': mapping, 'rows': rows}
        assert len(rows) == 57
        assert all(row['norm_F_2'] < .1 for row in rows)
        assert max(row['independent_scipy_prior_difference_fro'] for row in rows) < 1e-11
    output = HERE / 'gain_table.json'
    output.write_text(json.dumps(table, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({label: {
        'baseline': model['baseline_summary'], 'all_registered': model['registered_modes_summary']}
        for label, model in table['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
