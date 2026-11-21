"""Reproduce distribution and scalar estimation checks; Python 3.11+, stdlib only."""

import json
import math
from pathlib import Path
from statistics import NormalDist


N = NormalDist()
ROOT2 = math.sqrt(2.0)


def q(u):
    return 0.5 * N.inv_cdf((1.0 + u) / 2.0) ** 2


def sf(x):
    return 0.5 * math.erfc(x / ROOT2)


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(math.tau)


def mean(xs):
    return math.fsum(xs) / len(xs)


def decreasing_root(f, target):
    lo, hi = 0.0, 1.0
    while f(hi) > target:
        hi *= 2.0
    for _ in range(38):
        mid = (lo + hi) / 2.0
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def soft_moment(signal, variance, threshold):
    """E[(signal - soft(signal + N(0, variance), threshold))**2]."""
    s = math.sqrt(variance)
    a, b = (threshold - signal) / s, (-threshold - signal) / s
    right, left = sf(a), sf(-b)
    return ((variance + threshold**2) * (right + left)
            + variance * (a * phi(a) - b * phi(b))
            - 2.0 * threshold * s * (phi(a) + phi(b))
            + signal**2 * (1.0 - right - left))


def distribution_checks():
    delta, bmax2 = q(0.45), q(0.55)
    gamma, lam, cutoff, ratio = 1.1, 0.5, 0.15, 14 / 13
    probs = []
    for j in range(100):
        lower, upper = (lam * gamma**i - lam for i in (j, j + 1))
        probs.append(math.erfc(math.sqrt(lower)) - math.erfc(math.sqrt(upper)))
    assert abs(math.fsum(probs) - 1.0) < 1e-12
    entropy = -math.fsum(p * math.log(p) for p in probs if p)

    # The iid squared-normal difference equals a product of standard normals.
    h = 10.0 / 40000
    event = [2.0 * phi(0.0)] + [2.0 * phi(i*h) * math.erf(delta / (ROOT2*i*h))
                              for i in range(1, 40001)]
    near_probability = h / 3 * (event[0] + event[-1] + 4 * sum(event[1:-1:2])
                                 + 2 * sum(event[2:-1:2]))
    pmax = gamma * (lam + gamma**2 * cutoff**2 / delta)
    gap = delta - (gamma - 1) * bmax2
    c2 = max(math.hypot(1, gamma)/cutoff,
             pmax/(lam*gap) * math.sqrt(2 + 2*(gamma*cutoff/lam)**2))
    c1 = 1 / lam + ratio * math.sqrt(bmax2) / (2 * math.sqrt(lam)) * c2
    cx = math.sqrt(1 + bmax2 / (4 * lam)) + (lam + bmax2) * c2
    bound = math.hypot(c2, c1, cx)
    density_min = math.exp(-0.75) / math.sqrt(math.pi * 0.75)
    van_trees_coeff = 0.875 * density_min**2 * 0.75**2 / (0.5 + math.pi**2)
    return dict(delta=delta, smaller_squared_amplitude_cap=bmax2, bin_entropy_nats=entropy,
                lc_mi_exact_nats=entropy+math.log(20), lc_mi_input_upper_nats=entropy+math.log(40),
                lc_vs_iid_tv=1 - math.fsum(p*p for p in probs)/20,
                iid_near_probability=near_probability, hard_gap_kl_floor_nats=-math.log1p(-near_probability),
                hard_gap_kl_floor_bits=-math.log1p(-near_probability)/math.log(2),
                lc_left_inverse_row_bounds=[cx,c1,c2], lc_left_inverse_frobenius_bound=bound,
                lc_min_singular_lower=1/bound, van_trees_coefficient_lower=van_trees_coeff)


def scalar_check(n):
    quantiles = [(i + 0.5) / n for i in range(n)]
    coeffs = [0.5 + q(u) for u in quantiles]
    s0, s1 = 0.5, 0.58 / 1.12**2
    v = s0 + s1
    base = s0 * s1 / v
    budget = 0.02 * base
    coupled, iid = [], []
    for u, c0 in zip(quantiles, coeffs):
        for s, c_iid in zip(quantiles, coeffs):
            c_coupled = 0.5 + q((u + (0.45 + 0.1*s)) % 1.0)
            coupled.append(((s1*c0+s0*c_coupled)**2 / v**2, (c_coupled-c0)**2))
            iid.append(((s1*c0+s0*c_iid)**2 / v**2, (c_iid-c0)**2))

    expected_gain2 = mean([c/d2 for c,d2 in coupled])
    conservative_t = decreasing_root(lambda t: 5.396**2 * soft_moment(0, v, t), budget)
    exact_t = decreasing_root(lambda t: expected_gain2 * soft_moment(0, v, t), budget)

    # A common regularization family with independent budget calibration for both laws.
    def ridge_health(pairs, rho):
        return v * mean([c*d2/(d2+rho)**2 for c,d2 in pairs])

    def ridge_stats(pairs):
        rho = decreasing_root(lambda r: ridge_health(pairs, r), budget)
        bias2 = mean([c*(rho/(d2+rho))**2 for c,d2 in pairs])
        return rho, ridge_health(pairs, rho), bias2

    iid_ridge = ridge_stats(iid)
    coupled_ridge = ridge_stats(coupled)
    rows = []
    for limit in (0.0, 1.0, math.sqrt(10.0), 8.0, 16.0):
        risk = base + mean([c/d2 * soft_moment(math.sqrt(d2)*limit, v, exact_t) for c,d2 in coupled])
        rows.append(dict(nuisance_bound=limit, coupled_soft=risk,
                         iid_ridge=base+iid_ridge[1]+limit**2*iid_ridge[2],
                         coupled_ridge=base+coupled_ridge[1]+limit**2*coupled_ridge[2]))

    assert abs(rows[0]["coupled_soft"] - base - budget) < 1e-10
    for rho, health, bias in (iid_ridge, coupled_ridge):
        assert rho > 0 and 0 < health <= budget + 1e-12 and bias > 0
    assert all(b["coupled_soft"] >= a["coupled_soft"] for a,b in zip(rows, rows[1:]))
    return dict(midpoints_per_axis=n, mean_gain_squared_coupled=expected_gain2,
                mean_noise_variance=base, difference_noise_variance=v, healthy_increment_budget=budget,
                conservative_threshold=conservative_t, quadrature_threshold=exact_t,
                conservative_risk_upper=base + 5.396**2 * (v+2*conservative_t*math.sqrt(2*v/math.pi)+conservative_t**2),
                iid_ridge_lambda=iid_ridge[0], coupled_ridge_lambda=coupled_ridge[0], rows=rows)


if __name__ == "__main__":
    for variance in (0.5, 1.7):
        for signal in (0, 1, 5):
            assert abs(soft_moment(signal, variance, 0) - variance) < 1e-12
    data = dict(distribution=distribution_checks(), scalar=[scalar_check(n) for n in (128, 256, 512)])
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(out.name, "written")
