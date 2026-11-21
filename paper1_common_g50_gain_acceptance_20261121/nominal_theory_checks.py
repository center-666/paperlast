"""Deterministic checks of aggregation loss, nominal costs and calibration rank.

No attack Monte Carlo, and no claim about covariance after endogenous gating.
"""
from pathlib import Path
import json
import numpy as np
from design_common_g import solve_mode
from run_common_g_50 import load_common_g
from calibration_rule import path_fwer_cutoff

HERE = Path(__file__).resolve().parent


def nominal_covariance_path(model, mask):
    """True healthy covariance for a FIXED deterministic active-set protocol.

    Initial true error covariance is I, not the per-mode nominal DARE P.
    All noises are independent of fixed gains; no current-data selection here.
    """
    indices = model.gain_data['global_mask_to_local_row'][mask]
    B, F, W = [], [], []
    for i, index in enumerate(indices):
        row = model.gain_data['rows'][index]
        active = row['active_neighbors']
        K, G = np.asarray(row['K']), np.asarray(row['G'])
        b = np.eye(2) - K @ model.H[i] - G @ model.H[active].sum(axis=0)
        B.append(b)
        F.append(b @ model.A)
        W.append(b @ model.Q @ b.T + K @ model.R[i] @ K.T
                 + G @ model.R[active].sum(axis=0) @ G.T)
    F, W = np.asarray(F), np.asarray(W)
    covariance = np.broadcast_to(np.eye(2), (50, 2, 2)).copy()
    traces = [np.trace(covariance, axis1=1, axis2=2)]
    for k in range(1, 150):
        covariance = F @ covariance @ F.transpose(0, 2, 1) + W
        traces.append(np.trace(covariance, axis1=1, axis2=2))
    traces = np.asarray(traces)
    return {'network_mse_post20': float(traces[20:].mean()),
            'affected_mse_post20': float(traces[20:, model.receiver_ids].mean()),
            'network_mse_all150': float(traces.mean())}


def main():
    out = {'scope': 'Deterministic fixed-subset Gaussian design checks, not data-gated covariance or watermarked security performance.',
           'models': {}}
    for label in ['unstable', 'stable']:
        model = load_common_g(label)
        min_gap, max_identity, relative_information_loss = np.inf, 0.0, []
        for row in model.gain_data['rows']:
            i, active = row['receiver'], row['active_neighbors']
            meanH = model.H[active].mean(axis=0)
            stackedJ = 2 * model.H[i].T @ model.H[i]
            for j in active:
                stackedJ += 2 * model.H[j].T @ model.H[j]
            aggregateJ = np.asarray(row['J'])
            spread = sum((2 * (model.H[j] - meanH).T @ (model.H[j] - meanH)
                          for j in active), start=np.zeros((2, 2)))
            gap = stackedJ - aggregateJ
            min_gap = min(min_gap, float(np.linalg.eigvalsh(gap).min()))
            max_identity = max(max_identity, float(np.max(np.abs(gap - spread))))
            relative_information_loss.append(float(np.trace(gap) / np.trace(stackedJ)))
        assert min_gap >= -1e-11 and max_identity < 1e-11
        # Edge case for the general theorem: empty neighbor aggregate must not
        # leave a zero covariance block in an ordinary inverse.
        local_only = [solve_mode(model.A, model.Q, model.H, model.R, i, []) for i in range(50)]
        assert all(np.all(np.asarray(row['G']) == 0) for row in local_only)
        assert max(row['norm_F_2'] for row in local_only) < .5 * np.linalg.norm(model.A, 2)
        base = nominal_covariance_path(model, 0)
        drop = nominal_covariance_path(model, 63)
        out['models'][label] = {
            'information_loss_identity': 'J_stacked-J_aggregate=2 sum_j (H_j-H_mean)^T(H_j-H_mean) >= 0',
            'minimum_information_gap_eigenvalue': min_gap,
            'maximum_information_gap_identity_absolute_error': max_identity,
            'trace_relative_information_loss_range': [min(relative_information_loss), max(relative_information_loss)],
            'additional_local_only_designs_checked': 50,
            'maximum_local_only_F_norm': max(row['norm_F_2'] for row in local_only),
            'general_analytic_F_bound': float(.5 * np.linalg.norm(model.A, 2)),
            'fixed_baseline_healthy_covariance': base,
            'fixed_drop_known_six_healthy_covariance': drop,
            'fixed_drop_known_six_relative_healthy_cost_network_post20': drop['network_mse_post20'] / base['network_mse_post20'] - 1,
            'fixed_drop_known_six_relative_healthy_cost_affected_post20': drop['affected_mse_post20'] / base['affected_mse_post20'] - 1,
            'warning': 'Known-six fixed dropping rejects every registered attack; this candidate-support knowledge does not model unknown attack locations.'}
    rule = path_fwer_cutoff(np.arange(512), .05)
    assert rule['one_based_rank'] == 488 and rule['threshold'] == 487
    assert rule['marginal_path_fwer_bound'] == 25 / 513
    # Enumerate all placements of a fresh score in a strictly ordered pool:
    # this directly checks the exchangeability rank statement used in proof.
    exceedances = 0
    for held_out in range(513):
        calib = np.delete(np.arange(513), held_out)
        cutoff = path_fwer_cutoff(calib, .05)['threshold']
        exceedances += held_out > cutoff
    assert exceedances == 25
    out['future_calibration_rank_rule'] = rule
    out['future_calibration_rank_rule']['enumerated_fresh_rank_placements'] = 513
    out['future_calibration_rank_rule']['strict_exceedance_placements'] = exceedances
    (HERE / 'nominal_theory_checks.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
