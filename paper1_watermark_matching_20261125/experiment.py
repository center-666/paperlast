"""Explore receiver use of the realized watermark on the frozen v3 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
V3 = HERE.parent / 'paper1_sscd_commonrandom_20261122'
sys.path.insert(0, str(V3))
from run_crn_development import CANDIDATES, SCENARIOS_ANGLES, SCENARIOS_START, load_backend
from watson_crn import (
    beta_from_common, make_common_draws, theoretical_zeta_mi, watermark_attack_from_beta,
)

WINDOW = 8
DURATION = 60
BATCH = 32
THETA = {'iid': 0.0, 'negative': CANDIDATES['neg_boundary']}
ARMS = ('A', 'B', 'C', 'D', 'C_ref', 'D_ref')
ARM_CUTOFF = ('KL', 'KL', 'iid', 'negative', 'iid', 'negative')
CONTRASTS = {
    'KL_negative_minus_iid': (-1, 1, 0, 0, 0, 0),
    'matched_negative_minus_iid': (0, 0, -1, 1, 0, 0),
    'interaction': (1, -1, -1, 1, 0, 0),
    'matched_minus_KL_iid': (-1, 0, 1, 0, 0, 0),
    'matched_minus_KL_negative': (0, -1, 0, 1, 0, 0),
    'aligned_minus_reference_iid': (0, 0, 1, 0, -1, 0),
    'aligned_minus_reference_negative': (0, 0, 0, 1, 0, -1),
    'alignment_interaction': (0, 0, -1, 1, 1, -1),
}


def seed(root, split, model, scenario, chunk, stream):
    return np.random.SeedSequence([root, split, model, scenario, chunk, stream])


def draws_for(root, split, model, scenario, chunk, stream, count, config):
    seeds = [seed(root, split, model, scenario, chunk, stream + j) for j in range(3)]
    return make_common_draws(count, config.T, len(config.links), *seeds)


def matrices(beta):
    return 0.5 * np.eye(2) + beta[..., :, None] * beta[..., None, :]


def parity_whitening(model):
    """Whiten the row space of the four overlapping source parity residuals."""
    m = model.m
    powers = np.stack([np.linalg.matrix_power(model.A, t) for t in range(WINDOW)])
    observations = model.H[model.links[:, 1], None] @ powers
    whitened = []
    for edge, design in enumerate(observations):
        sender = model.links[edge, 1]
        observability = design.reshape(WINDOW * m, model.n)
        parity = null_space(observability.T).T
        source = model.Z[edge]
        stacked = np.vstack([
            np.pad(source, ((0, 0), (j * m, WINDOW * m - j * m - source.shape[1])))
            for j in range(4)
        ])
        if np.linalg.matrix_rank(stacked) != parity.shape[0]:
            raise AssertionError('Eight-packet parity must have the legacy stacked row space')
        np.testing.assert_allclose(stacked @ observability, 0.0, atol=2e-12)
        np.testing.assert_allclose(stacked @ parity.T @ parity, stacked, atol=2e-12)

        process_map = np.zeros((WINDOW * m, (WINDOW - 1) * model.n))
        for t in range(1, WINDOW):
            for j in range(1, t + 1):
                process_map[t * m:(t + 1) * m, (j - 1) * model.n:j * model.n] = design[t - j]
        noise = np.kron(np.eye(WINDOW), model.R[sender])
        noise += process_map @ np.kron(np.eye(WINDOW - 1), model.Q) @ process_map.T
        whiten = np.linalg.solve(np.linalg.cholesky(parity @ noise @ parity.T), parity)
        np.testing.assert_allclose(whiten @ noise @ whiten.T, np.eye(len(whiten)), atol=2e-12)
        whitened.append(whiten)
    linear = np.stack(whitened)
    blocks = linear.reshape(len(linear), -1, WINDOW, m).transpose(0, 2, 1, 3)
    return linear, blocks, observations


def suffix_templates(key_window, blocks, observations):
    # Each candidate attack start q produces M_t H A**(t-q) xi for t >= q.
    per_packet = np.einsum('ewrm,bewmn,ewnd->berwd',
                           blocks, key_window, observations, optimize=True)
    # Invertible A makes H A**j and H A**(j-q) equivalent for each suffix q.
    reverse_sum = np.cumsum(per_packet[:, :, :, ::-1, :], axis=3)
    return reverse_sum[:, :, :, ::-1, :].transpose(0, 1, 3, 2, 4)


def projection_energy(residual, design):
    """Squared projection on each candidate's two-column template span."""
    u, v = design[..., 0], design[..., 1]
    uu = np.einsum('...d,...d->...', u, u)
    vv = np.einsum('...d,...d->...', v, v)
    scale = np.maximum(uu, vv)
    pivot = np.where((uu >= vv)[..., None], u, v)
    other = np.where((uu >= vv)[..., None], v, u)
    unit = pivot / np.sqrt(np.where(scale > 1e-24, scale, np.inf))[..., None]
    perp = other - np.einsum('...i,...i->...', unit, other)[..., None] * unit
    perp_norm2 = np.einsum('...d,...d->...', perp, perp)
    first = np.einsum('...d,...qd->...q', residual, unit)
    second = np.einsum('...d,...qd->...q', residual, perp)
    denom = np.where(perp_norm2 > np.maximum(1e-24, scale * 1e-24), perp_norm2, np.inf)
    return first**2 + second**2 / denom


def window_residual(packets, linear):
    paths, _, edges, _ = packets.shape
    flat = packets.transpose(0, 2, 1, 3).reshape(paths, edges, -1)
    return np.einsum('edm,bem->bed', linear, flat)


def matched_scores(packets, keys, operator, *, stop=None):
    linear, blocks, observations = operator
    output = np.zeros(packets.shape[:3])
    t_stop = packets.shape[1] if stop is None else stop
    packets_window = np.lib.stride_tricks.sliding_window_view(
        packets[:, :t_stop], WINDOW, axis=1
    ).transpose(0, 1, 2, 4, 3)
    keys_window = np.lib.stride_tricks.sliding_window_view(
        keys[:, :t_stop], WINDOW, axis=1
    ).transpose(0, 1, 2, 5, 3, 4)
    residual = np.einsum(
        'edm,btem->bted', linear,
        packets_window.reshape(*packets_window.shape[:3], -1), optimize=True,
    )
    per_packet = np.einsum(
        'ewrm,btewmn,ewnd->bterwd', blocks, keys_window, observations, optimize=True,
    )
    reverse_sum = np.cumsum(per_packet[:, :, :, :, ::-1, :], axis=4)
    templates = reverse_sum[:, :, :, :, ::-1, :].transpose(0, 1, 2, 4, 3, 5)
    first_time = WINDOW - 1
    output[:, first_time:t_stop] = projection_energy(
        residual[:, first_time - (WINDOW - 1):],
        templates[:, first_time - (WINDOW - 1):],
    ).max(axis=3)
    return output


def check_projection(model, operator):
    """Check the optimized score against explicit templates and least squares."""
    linear, blocks, observations = operator
    if np.linalg.matrix_rank(model.A) != model.n:
        raise AssertionError('Suffix reparameterization requires invertible dynamics')
    rng = np.random.default_rng(19277)
    packets = rng.normal(size=(3, WINDOW + 2, len(model.links), model.m))
    keys = matrices(rng.normal(size=packets.shape))
    t = WINDOW
    recent = slice(t - WINDOW + 1, t + 1)
    actual = matched_scores(packets, keys, operator, stop=t + 1)[:, t]
    for b in range(len(packets)):
        for e in range(len(model.links)):
            z = linear[e] @ packets[b, recent, e].reshape(-1)
            expected = []
            for q in range(WINDOW):
                template = np.zeros((WINDOW, model.m, model.n))
                template[q:] = keys[b, recent, e][q:] @ observations[e, :WINDOW - q]
                u = linear[e] @ template.reshape(WINDOW * model.m, model.n)
                projection = u @ np.linalg.lstsq(u, z, rcond=None)[0]
                expected.append(projection @ projection)
            np.testing.assert_allclose(actual[b, e], max(expected), atol=1e-11)

    key_window = keys[:, recent].swapaxes(1, 2)
    templates = suffix_templates(key_window, blocks, observations)
    residual = window_residual(packets[:, recent], linear)
    expected_one = []
    for b in range(len(packets)):
        for e in range(len(model.links)):
            z = residual[b, e]
            u = templates[b, e, 0]
            projection = u @ np.linalg.lstsq(u, z, rcond=None)[0]
            expected_one.append(projection @ projection)
    np.testing.assert_allclose(
        projection_energy(residual, templates[:, :, :1]).squeeze(axis=2),
        np.asarray(expected_one).reshape(len(packets), len(model.links)),
        atol=1e-11,
    )

    packets[:, t + 1:] += 1e3
    keys[:, t + 1:] *= 1e3
    np.testing.assert_array_equal(matched_scores(packets, keys, operator, stop=t + 1)[:, t], actual)


def health_maxima(model, operator, physical_paths, legacy, *, root, split, mi, n):
    records = []
    for chunk, first in enumerate(range(0, n, BATCH)):
        count = min(BATCH, n - first)
        phy = physical_paths(model, count, seed(root, split, mi, 0, chunk, 0))
        packets = phy.y[:, :, model.links[:, 1]]
        null_delta = np.zeros_like(packets)
        maxima = [legacy(model, phy, null_delta)[:, WINDOW:].max(axis=(1, 2))]
        for stream in (1, 4):
            draws = draws_for(root, split, mi, 0, chunk, stream, count, model)
            for theta in THETA.values():
                keys = matrices(beta_from_common(draws, theta))
                maxima.append(matched_scores(packets, keys, operator)[:, WINDOW:].max(axis=(1, 2)))
        records.append(np.stack(maxima, axis=1))
    return np.concatenate(records)


def alarm_metrics(score, cutoff, onset):
    alarm = score > cutoff
    in_attack = alarm[:, onset:onset + DURATION].any(axis=2)
    delay = np.where(in_attack.any(axis=1), in_attack.argmax(axis=1), DURATION)
    return {
        'delay': delay,
        'prior_alarm': alarm[:, WINDOW:onset].any(axis=(1, 2)),
        'eligible_rate': alarm[:, onset + WINDOW - 1:onset + DURATION].mean(axis=(1, 2)),
    }


def attack_metrics(model, operator, physical_paths, legacy, raw, onset, *, root, mi,
                   scenario, n, thresholds):
    records = []
    linear, blocks, observations = operator
    for chunk, first in enumerate(range(0, n, BATCH)):
        count = min(BATCH, n - first)
        phy = physical_paths(model, count, seed(root, 3, mi, scenario, chunk, 0))
        actual = draws_for(root, 3, mi, scenario, chunk, 1, count, model)
        reference = draws_for(root, 3, mi, scenario, chunk, 4, count, model)
        metrics = {}
        for i, (name, theta) in enumerate(THETA.items()):
            beta = beta_from_common(actual, theta)
            delta = watermark_attack_from_beta(raw, beta[:, onset:onset + DURATION],
                                               T=model.T, offset=onset)
            packets = phy.y[:, :, model.links[:, 1]] + delta
            actual_key = matrices(beta)
            reference_key = matrices(beta_from_common(reference, theta))
            scores = [legacy(model, phy, delta)]
            scores.extend(matched_scores(packets, keys, operator, stop=onset + DURATION)
                          for keys in (actual_key, reference_key))
            arms = (ARMS[i], ARMS[2 + i], ARMS[4 + i])
            cutoffs = (thresholds['KL'], thresholds[name], thresholds[name])
            for arm, score, alarm_cutoff in zip(arms, scores, cutoffs):
                metrics[arm] = alarm_metrics(score, alarm_cutoff, onset)

            signal_energy = []
            reference_projection = []
            for attack_age in (1, 2, WINDOW):
                t = onset + attack_age - 1
                recent = slice(t - WINDOW + 1, t + 1)
                signal = window_residual(delta[:, recent], linear)
                signal_energy.append((signal**2).sum(axis=-1).mean(axis=1))
                true_start = WINDOW - attack_age
                geometry = []
                for keys in (actual_key, reference_key):
                    templates = suffix_templates(keys[:, recent].swapaxes(1, 2), blocks, observations)
                    projection = projection_energy(signal, templates[:, :, true_start:true_start + 1])
                    geometry.append(projection.squeeze(axis=2).mean(axis=1))
                np.testing.assert_allclose(geometry[0], signal_energy[-1], atol=1e-10, rtol=1e-11)
                reference_projection.append(geometry[1])
            metrics[name] = {
                'signal_energy': np.stack(signal_energy, axis=1),
                'reference_projection_energy': np.stack(reference_projection, axis=1),
            }
        records.append({
            field: np.stack([metrics[arm][field] for arm in ARMS], axis=1)
            for field in ('delay', 'prior_alarm', 'eligible_rate')
        } | {
            field: np.stack([metrics[name][field] for name in THETA], axis=1)
            for field in ('signal_energy', 'reference_projection_energy')
        })
    return {field: np.concatenate([row[field] for row in records]) for field in records[0]}


def path_rate(hit):
    count = int(hit.sum())
    interval = binomtest(count, len(hit)).proportion_ci(method='exact')
    return {'n': len(hit), 'alarms': count, 'rate': float(hit.mean()),
            'ci95_exact': [interval.low, interval.high]}


def paired_summary(values, contrast_matrix):
    """Equal-scenario means and path-level SE conditional on calibration."""
    contrasts = [np.einsum('pa...,ca->pc...', row, contrast_matrix) for row in values]
    mean = np.mean([row.mean(axis=0) for row in contrasts], axis=0)
    se = np.sqrt(np.sum([row.var(axis=0, ddof=1) / len(row) for row in contrasts], axis=0)) / len(values)
    return [{'mean': m.tolist(), 'se': s.tolist(), 'ci95': [lo.tolist(), hi.tolist()]}
            for m, s, lo, hi in zip(mean, se, mean - 1.96 * se, mean + 1.96 * se)]


def summarize(scenarios):
    indicators = [(r['delay'][..., None] < np.arange(1, WINDOW + 1)).astype(float)
                  for r in scenarios]
    measurements = {
        'early_alarm_cdf_h1_to_h8': indicators,
        'early_alarm_with_no_prior_alarm_cdf': [x * (~r['prior_alarm'][..., None])
                                              for x, r in zip(indicators, scenarios)],
        'restricted_delay_8': [np.minimum(r['delay'], WINDOW).astype(float) for r in scenarios],
        'prior_alarm': [r['prior_alarm'].astype(float) for r in scenarios],
        'eligible_rate': [r['eligible_rate'] for r in scenarios],
    }
    out = {}
    for key, rows in measurements.items():
        result = paired_summary(rows, np.array(list(CONTRASTS.values())))
        arm_mean = np.mean([r.mean(axis=0) for r in rows], axis=0)
        out[key] = {
            'arm_mean': dict(zip(ARMS, arm_mean.tolist())),
            'contrasts': dict(zip(CONTRASTS, result)),
        }
    for field in ('signal_energy', 'reference_projection_energy'):
        rows = [r[field] for r in scenarios]
        averages = np.mean([r.mean(axis=0) for r in rows], axis=0)
        out[field] = {
            'mean_h1_h2_h8': dict(zip(THETA, averages.tolist())),
            'negative_minus_iid': paired_summary(rows, np.array([[-1, 1]]))[0],
        }
    return out


def run(args):
    lock = json.loads((V3 / 'crn_locked_selection.json').read_text(encoding='utf-8'))
    if lock['selection'] != 'neg_boundary' or lock['theta_locked'] != THETA['negative']:
        raise ValueError('Saved v3 selection changed')
    if not args.smoke and ((args.output / 'results.json').exists() or (args.output / 'paths.npz').exists()):
        raise FileExistsError(f'Existing results in {args.output}')
    old, load, physical_paths, _, cutoff = load_backend()
    from replay import scores as legacy

    audit = old.validate_fixtures()
    audit.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (V3 / 'watson_crn.py', V3 / 'crn_locked_selection.json', Path(__file__))})
    result = {
        'source_reference': '8953e1dd', 'source_sha256': audit, 'seed': args.seed,
        'numpy': np.__version__, 'batch_size': BATCH, 'theta': THETA,
        'observation_window': WINDOW, 'first_scored_time': WINDOW,
        'target_path_FWER': 0.05, 'primary_h': 2,
        'arm_order': ARMS, 'contrast_order': list(CONTRASTS),
        'calibration_paths_per_model': args.calibration,
        'heldout_paths_per_model': args.healthy,
        'attack_test_paths_per_scenario': args.paths, 'models': {},
    }
    result['two_packet_energy_coefficient'] = {
        k: 1.5 - theoretical_zeta_mi(v)[0] / 2 for k, v in THETA.items()
    }
    arrays = {}
    for mi, label in enumerate(('unstable', 'stable')):
        model = load(label)
        operator = parity_whitening(model)
        check_projection(model, operator)
        cal = health_maxima(model, operator, physical_paths, legacy, root=args.seed,
                            split=1, mi=mi, n=args.calibration)
        healthy = health_maxima(model, operator, physical_paths, legacy, root=args.seed,
                                split=2, mi=mi, n=args.healthy)
        names = ('KL', *THETA)
        rules = {name: cutoff(cal[:, i], 0.05) for i, name in enumerate(names)}
        thresholds = {name: rule['threshold'] for name, rule in rules.items()}
        fwer = dict(zip(ARMS, [path_rate(healthy[:, col] > thresholds[name])
                              for col, name in zip((0, 0, 1, 2, 3, 4), ARM_CUTOFF)]))
        arrays[label + '__calibration_max'] = cal
        arrays[label + '__healthy_max'] = healthy
        print(f'{label}: calibration and healthy holdout complete', flush=True)
        scenarios = {}
        sequence = []
        angles = SCENARIOS_ANGLES[:1] if args.smoke else SCENARIOS_ANGLES
        starts = SCENARIOS_START[:1] if args.smoke else SCENARIOS_START
        for ai, angle in enumerate(angles):
            raw, _ = old.rotated_raw_attack(model, angle)
            for oi, onset in enumerate(starts):
                key = f'{label}_rot{angle:+.0f}_onset{onset}'
                record = attack_metrics(model, operator, physical_paths, legacy, raw[50:110], onset,
                                        root=args.seed, mi=mi, scenario=1 + ai * 2 + oi,
                                        n=args.paths, thresholds=thresholds)
                sequence.append(record)
                arrays.update({key + '__' + k: v for k, v in record.items()})
                scenarios[key] = summarize([record])
                print(f'{key}: paired test complete', flush=True)
        result['models'][label] = {
            'calibration': rules,
            'healthy_FWER': fwer,
            'summary': summarize(sequence),
            'scenarios': scenarios,
        }

    if args.smoke:
        print('Smoke checks passed for both models; no files written.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'paths.npz'
    np.savez_compressed(archive, **arrays)
    result['path_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / 'results.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=202611250617)
    parser.add_argument('--calibration', type=int, default=2048)
    parser.add_argument('--healthy', type=int, default=2048)
    parser.add_argument('--paths', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.calibration, args.healthy, args.paths = 32, 32, 8
        args.seed += 1
    if min(args.calibration, args.healthy, args.paths) < 20 and not args.smoke:
        parser.error('Use at least 20 paths per split and scenario')
    run(args)


if __name__ == '__main__':
    main()
