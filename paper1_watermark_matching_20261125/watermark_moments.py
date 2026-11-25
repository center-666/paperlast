"""Exact second-moment decomposition of the matched detector inputs."""

import json

import numpy as np

from experiment import (
    HERE, SCENARIOS_ANGLES, THETA, WINDOW, load_backend, parity_whitening,
    theoretical_zeta_mi,
)


def energy_terms(blocks, raw):
    """Return the theta-invariant energy and coefficients of positive lags."""
    age = len(raw)
    blocks = blocks[:, -age:]
    raw = raw.swapaxes(0, 1)
    directions = np.array([[[1, 0], [0, -1]], [[0, 1], [1, 0]]])
    deterministic = np.einsum('etdm,etm->ed', blocks, raw)
    radial = np.einsum('etdm,etm->etd', blocks, raw)
    angular = np.einsum('etdm,imn,etn->etid', blocks, directions, raw)
    energy = np.sum(deterministic**2, axis=1)
    energy += 0.25 * (radial**2).sum(axis=(1, 2))
    energy += 0.25 * (angular**2).sum(axis=(1, 2, 3))
    lags = np.array([
        0.25 * np.mean(np.sum(angular[:, lag:] * angular[:, :-lag], axis=(1, 2, 3)))
        for lag in range(1, age)
    ])
    return float(energy.mean()), lags


def main():
    old, load, _, _, _ = load_backend()
    zeta = theoretical_zeta_mi(THETA['negative'])[0]
    sample = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['models']
    out = {'negative_zeta': zeta, 'models': {}}
    for label in ('unstable', 'stable'):
        model = load(label)
        _, blocks, _ = parity_whitening(model)
        raw = [old.rotated_raw_attack(model, angle)[0][50:58] for angle in SCENARIOS_ANGLES]
        by_age = {}
        for index, age in enumerate((1, 2, WINDOW)):
            terms = [energy_terms(blocks, attack[:age]) for attack in raw]
            iid = float(np.mean([baseline for baseline, _ in terms]))
            coefficients = np.mean([lags for _, lags in terms], axis=0)
            increments = coefficients * zeta**np.arange(1, age)
            negative = iid + float(increments.sum())
            monte_carlo = sample[label]['summary']['signal_energy']['negative_minus_iid']
            sample_interval = [side[index] for side in monte_carlo['ci95']]
            by_age[age] = {
                'iid_mean': iid,
                'negative_mean': negative,
                'negative_minus_iid': float(increments.sum()),
                'relative_increase': float(increments.sum() / iid),
                'negative_minus_iid_by_lag': increments.tolist(),
                'monte_carlo_paired_ci95': sample_interval,
            }
        out['models'][label] = by_age
    (HERE / 'watermark_moments.json').write_text(
        json.dumps(out, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n'
    )
    for label, by_age in out['models'].items():
        for age, record in by_age.items():
            print(label, age, round(record['iid_mean'], 3), round(record['negative_mean'], 3),
                  np.round(record['negative_minus_iid_by_lag'], 3).tolist())


if __name__ == '__main__':
    main()
