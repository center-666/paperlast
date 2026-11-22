"""Exact no-gate watermark-to-state-risk bridge for the new common-G table.

The calculation uses the frozen 50-node physical model and current-mode
steady gains. It is a deterministic moment calculation, not a Monte Carlo
experiment and not the endogenous-gate final MSE. Independent source chains
retain separate covariance terms; deterministic first moments retain all
cross-source terms, including the two attacks received by node 17.
"""
from pathlib import Path
import json
import numpy as np

from run_common_g_50 import load_common_g
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper1_SSCD_review_20261121'))
import sscd

HERE = Path(__file__).resolve().parent
START, STOP = 20, 150
ATTACK_START, ATTACK_STOP = 50, 110
LAMBDA, QVAR = .5, .5


def full_covariance_blocks(ds, zeta):
    """Return covariance of [beta_l beta_l' d_l] for one source chain."""
    ds = np.asarray(ds, float)
    L, m = ds.shape
    out = np.zeros((L * m, L * m))
    for ell, d in enumerate(ds):
        sl = slice(m * ell, m * (ell + 1))
        out[sl, sl] = QVAR ** 2 * (d @ d * np.eye(m) + np.outer(d, d))
        for r in range(ell + 1, L):
            sr = slice(m * r, m * (r + 1))
            eta = QVAR ** 2 * m / (m + 2.) * zeta ** (r - ell)
            cross = eta * ((d @ ds[r]) * np.eye(m)
                           + np.outer(ds[r], d)
                           - (2. / m) * np.outer(d, ds[r]))
            out[sl, sr] = cross
            out[sr, sl] = cross.T
    return (out + out.T) / 2


def source_covariance_polynomial(T_blocks, K_blocks):
    """Coefficient matrices for one source covariance, powers zeta^h."""
    T = np.asarray(T_blocks, float)
    K = np.asarray(K_blocks, float)
    L = T.shape[0]
    Q = np.zeros((L, 2, 2))
    q2 = QVAR ** 2
    KT = np.einsum('lij,ljk->lik', K, T)
    tr_KKt = np.einsum('lij,lij->l', K, K)
    TtT = np.einsum('lki,lkj->lij', T, T)
    KTtKT = np.einsum('lki,lkj->lij', KT, KT)
    Q[0] = q2 * np.sum(tr_KKt[:, None, None] * TtT + KTtKT, axis=0)
    cross_constant = q2 * 2. / 4.  # q^2 m/(m+2), m=2
    for h in range(1, L):
        U, V = T[:-h], T[h:]
        Kl, Kr = K[:-h], K[h:]
        B = np.einsum('nki,nkj->nij', Kr, Kl)
        UV = np.einsum('nki,nkj->nij', U, V)
        UB = np.einsum('nki,nkj->nij', U, B)
        UBV = np.einsum('nij,njk->nik', UB, V)
        VB = np.einsum('nki,nkj->nij', V, B)
        VBU = np.einsum('nij,njk->nik', VB, U)
        symUV = .5 * (UV + UV.transpose(0, 2, 1))
        symUBV = .5 * (UBV + UBV.transpose(0, 2, 1))
        symVBU = .5 * (VBU + VBU.transpose(0, 2, 1))
        Q[h] = 2. * cross_constant * np.sum(
            np.trace(B, axis1=1, axis2=2)[:, None, None] * symUV
            + symUBV - symVBU, axis=0)
    return (Q + Q.transpose(0, 2, 1)) / 2


def state_risk_polynomial(model, horizon_start=START, horizon_stop=STOP):
    """Coefficient matrices Q_h so Q(zeta)=sum_h zeta^h Q_h."""
    H, A = model.H, model.A
    # All registered attack edges are admitted in the no-gate common-G mode.
    incoming = {i: [e for e, (receiver, _) in enumerate(model.links) if receiver == i]
               for i in model.receiver_ids}
    Q = np.zeros((ATTACK_STOP - ATTACK_START, 2, 2))
    mu = LAMBDA + QVAR
    weight = 1. / (50. * (horizon_stop - horizon_start))
    powers = [np.linalg.matrix_power(A, t) for t in range(ATTACK_STOP - ATTACK_START)]
    for receiver in model.receiver_ids:
        i = int(receiver)
        edges = incoming[i]
        # For each source edge, output blocks map its 60 watermark matrices to
        # each evaluation time. The same source's blocks share one chain.
        mean_by_time = np.zeros((horizon_stop - horizon_start, 2, 2))
        source_cov_by_time = []
        for edge in edges:
            sender = int(model.links[edge, 1])
            active = model.gain_data['rows'][model.gain_data['global_mask_to_local_row'][0][i]]['active_neighbors']
            G = model.G_lookup[0, i]
            # D[e,l,k] is the 2x2 transfer from delta_{sender,START+l}
            # to receiver state at k.
            T_blocks = [H[sender] @ powers[t] for t in range(len(powers))]
            K_blocks_by_time = []
            source_mean = np.zeros((horizon_stop - horizon_start, 2, 2))
            source_cov = np.zeros((horizon_stop - horizon_start,
                                   ATTACK_STOP - ATTACK_START, 2, 2))
            for out_idx, k in enumerate(range(horizon_start, horizon_stop)):
                Kblocks = []
                for t in range(len(powers)):
                    attack_time = ATTACK_START + t
                    if k < attack_time:
                        Kblocks.append(np.zeros((2, 2)))
                    else:
                        F = model.F_lookup[0, i]
                        Kblocks.append(np.linalg.matrix_power(F, k - attack_time) @ G)
                K_blocks_by_time.append(Kblocks)
                source_mean[out_idx] = mu * sum((K @ T for K, T in zip(Kblocks, T_blocks)),
                                                np.zeros((2, 2)))
                source_cov[out_idx] = source_covariance_polynomial(T_blocks, Kblocks)
            mean_by_time += source_mean
            source_cov_by_time.append(source_cov)
        for out_idx in range(horizon_stop - horizon_start):
            mean = mean_by_time[out_idx]
            Q[0] += weight * (mean.T @ mean)
            for source_cov in source_cov_by_time:
                Q += weight * source_cov[out_idx]
    return (Q + Q.transpose(0, 2, 1)) / 2


def state_risk_matrix(model, zeta, horizon_start=START, horizon_stop=STOP):
    coeff = state_risk_polynomial(model, horizon_start, horizon_stop)
    Q = np.zeros((2, 2))
    for h, Qh in enumerate(coeff):
        Q += (zeta ** h) * Qh
    return (Q + Q.T) / 2


def output(model, label):
    zetas = {
        'iid': 0.0,
        'negative': float(sscd.watson_stats(2, -1.315236976827179)['zeta']),
        'positive': float(sscd.watson_stats(2, 1.315236976827179)['zeta']),
    }
    matrices = {name: state_risk_matrix(model, z) for name, z in zetas.items()}
    xi = model.xi
    E = np.zeros((2, 2))
    for t in range(ATTACK_STOP - ATTACK_START):
        for _, sender in model.links:
            T = model.H[int(sender)] @ np.linalg.matrix_power(model.A, t)
            E += T.T @ T
    energy = float(xi @ E @ xi)
    # E-normalized worst-direction risk, useful when comparing attack geometry
    # independently of the historical dominant-mode xi.
    ev, V = np.linalg.eigh(E)
    Einv = (V / np.sqrt(ev)) @ V.T
    worst = {name: float(energy * np.linalg.eigvalsh(Einv @ Q @ Einv)[-1])
             for name, Q in matrices.items()}
    # Q(zeta) is polynomial in zeta. A symmetric finite difference at zero is
    # a diagnostic of the exact first-order cross-time contribution.
    coeff = state_risk_polynomial(model)
    Q1 = coeff[1]
    result = {
        'model': label,
        'scope': 'Exact no-gate common-G moment bridge; source chains independent, source means summed before squaring.',
        'horizon': [START, STOP],
        'evaluation_window': [START, STOP],
        'attack_interval': [ATTACK_START, ATTACK_STOP],
        'watermark': {'lambda': LAMBDA, 'coordinate_variance_q': QVAR,
                      'zeta_by_policy': zetas, 'mutual_information_nats_per_step': {
                          name: float(sscd.watson_stats(2, sscd.theta_from_zeta(2, z))['mi_nats'])
                          for name, z in zetas.items()}},
        'historical_attack_energy': energy,
        'risk_matrix_by_policy': {name: Q.tolist() for name, Q in matrices.items()},
        'historical_direction_risk': {name: float(xi @ Q @ xi) for name, Q in matrices.items()},
        'historical_direction_relative_to_iid': {
            name: float((xi @ Q @ xi) / (xi @ matrices['iid'] @ xi) - 1)
            for name, Q in matrices.items()},
        'energy_normalized_worst_direction_risk': worst,
        'energy_normalized_worst_direction_relative_to_iid': {
            name: float(value / worst['iid'] - 1) for name, value in worst.items()},
        'iid_directional_first_derivative': float(xi @ Q1 @ xi),
        'iid_worst_direction_first_derivative_diagnostic': float(
            np.linalg.eigvalsh(Einv @ Q1 @ Einv)[-1] * energy),
        'Q_first_derivative_matrix': Q1.tolist(),
        'claims_supported': [
            'The sign of each reported no-gate risk change is a deterministic moment result for the declared attack family.',
            'The mean response is zeta-invariant; zeta changes only the within-source covariance cross-time terms.'
        ],
        'claims_not_supported': [
            'Non-Gaussian plugin-KL detection power.',
            'Endogenous gate probability or final gated MSE.',
            'Unknown attack locations or arbitrary attack directions beyond the energy-normalized 2D shell.'
        ]
    }
    return result


def generic_moment_identity_check():
    """Cross-check optimized polynomial against the reviewed generic routine."""
    rng = np.random.default_rng(2026112209)
    T = rng.normal(size=(7, 2, 2))
    K = rng.normal(size=(7, 2, 2))
    coeff = source_covariance_polynomial(T, K)
    mean = (LAMBDA + QVAR) * sum((K[i] @ T[i] for i in range(7)), np.zeros((2, 2)))
    errors = []
    for zeta in [-.3122270913392132, 0., .31222709133921356]:
        ours = mean.T @ mean + sum((zeta ** h) * coeff[h] for h in range(len(coeff)))
        reference = sscd.common_g_risk_matrix(list(T), list(K), np.eye(2), zeta, QVAR, LAMBDA)
        errors.append(float(np.max(np.abs(ours - reference))))
    return {'max_absolute_difference': max(errors), 'tested_zeta_values': 3,
            'pass_tolerance': 1e-11}


def main():
    result = {'status': 'PASS', 'source': 'gain_table.json + physical_configuration.json',
              'generic_moment_identity_check': generic_moment_identity_check(), 'models': {}}
    assert result['generic_moment_identity_check']['max_absolute_difference'] < 1e-11
    for label in ['unstable', 'stable']:
        result['models'][label] = output(load_common_g(label), label)
    (HERE / 'common_g_watermark_risk.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
