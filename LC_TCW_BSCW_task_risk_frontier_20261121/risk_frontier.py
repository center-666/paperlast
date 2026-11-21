"""Reproduce the BSCW-alpha two-sample risk frontier.

Python 3.10+, numpy, scipy. Monte Carlo is over watermark pairs only;
conditional state/noise risks are evaluated analytically.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erfc, ndtri
from scipy.stats import t as student_t

LAMBDA_W = 0.5
RHO = 14.0 / 13.0
SIGMA = np.array([0.5, 0.5, 0.5, 0.58 / 1.12**2], dtype=float)
W = 1.0 / SIGMA
BX = np.array([0.0, 1.0, 0.0, 1.0])
PX = 1.0
HEALTH_BUDGET = 0.02
ATTACK_ENERGY = 10.0
N_REPLICATES = 15
N_PER_REPLICATE = 50_000


def qchi(u):
    """Quantile of 0.5 * chi-square(1)."""
    u = np.clip(np.asarray(u), 1e-14, 1.0 - 1e-14)
    return 0.5 * ndtri((1.0 + u) / 2.0) ** 2


def sample_iid(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, math.sqrt(0.5), size=(n, 2, 2))


def sample_bscw(n, seed, alpha):
    """BSCW-alpha: couple only squared b-amplitudes; signs remain iid."""
    rng = np.random.default_rng(seed)
    u0 = rng.random(n)
    shift = rng.uniform(alpha, 1.0 - alpha, n)
    u1 = (u0 + shift) % 1.0
    beta = np.empty((n, 2, 2), dtype=float)
    beta[:, :, 0] = rng.normal(0.0, math.sqrt(0.5), size=(n, 2))
    beta[:, 0, 1] = np.sqrt(qchi(u0)) * rng.choice([-1.0, 1.0], n)
    beta[:, 1, 1] = np.sqrt(qchi(u1)) * rng.choice([-1.0, 1.0], n)
    return beta


def sample_full_lctcw(n, seed, gamma=1.1, lam=0.5):
    """Reproduce the archived full LC-TCW two-time generator."""
    rng = np.random.default_rng(seed)

    # Draw the common geometric-bin index using the original a^2 law.
    x_raw = rng.normal(0.0, math.sqrt(0.5), n) ** 2
    j = np.floor(np.log1p(x_raw / lam) / math.log(gamma)).astype(int)
    lo = lam * np.power(gamma, j) - lam
    hi = lam * np.power(gamma, j + 1) - lam
    f_lo = 1.0 - erfc(np.sqrt(lo))
    f_hi = 1.0 - erfc(np.sqrt(hi))
    f0 = f_lo + rng.random(n) * (f_hi - f_lo)
    f1 = f_lo + rng.random(n) * (f_hi - f_lo)
    x0 = qchi(f0)
    x1 = qchi(f1)
    sign_a = rng.choice([-1.0, 1.0], n)
    a0 = sign_a * np.sqrt(x0)
    a1 = -sign_a * np.sqrt(x1)

    u0 = rng.random(n)
    shift = rng.uniform(0.45, 0.55, n)
    u1 = (u0 + shift) % 1.0
    sign_b = rng.choice([-1.0, 1.0], n)
    b0 = sign_b * np.sqrt(qchi(u0))
    b1 = sign_b * np.sqrt(qchi(u1))

    beta = np.empty((n, 2, 2), dtype=float)
    beta[:, 0, 0], beta[:, 1, 0] = a0, a1
    beta[:, 0, 1], beta[:, 1, 1] = b0, b1
    return beta


def build_block(beta):
    a0, a1 = beta[:, 0, 0], beta[:, 1, 0]
    b0, b1 = beta[:, 0, 1], beta[:, 1, 1]
    p0, p1 = LAMBDA_W + a0*a0, LAMBDA_W + a1*a1
    q0, q1 = a0*b0, a1*b1
    r0, r1 = LAMBDA_W + b0*b0, LAMBDA_W + b1*b1

    block = np.zeros((len(beta), 4, 3), dtype=float)
    block[:, 0, 1], block[:, 0, 2] = p0, q0
    block[:, 1, 0], block[:, 1, 1], block[:, 1, 2] = 1.0, q0, r0
    block[:, 2, 1], block[:, 2, 2] = p1, RHO*q1
    block[:, 3, 0], block[:, 3, 1], block[:, 3, 2] = 1.0, q1/RHO, r1
    return block


def prepare(block, px):
    """Precompute the state-orthogonal attack information eigendecomposition."""
    b = block[:, :, 0]
    attack = block[:, :, 1:]
    bw = b * W[None, :]
    p = np.sum(b * bw, axis=1)
    p0 = float(p[0])
    v = np.einsum("nr,nri->ni", bw, attack)
    a = v / p[:, None]

    attack_w = attack * W[None, :, None]
    gram = np.einsum("nri,nrj->nij", attack, attack_w)
    cmat = gram - np.einsum("ni,nj->nij", v, v) / p[:, None, None]
    cmat = 0.5 * (cmat + np.transpose(cmat, (0, 2, 1)))
    eigval, eigvec = np.linalg.eigh(cmat)
    eigval = np.maximum(eigval, 0.0)
    a_eig = np.einsum("ni,nij->nj", a, eigvec)
    kappa = px * p / (1.0 + px * p)
    h0 = px / (1.0 + px * p0)
    return eigval, eigvec, a_eig, kappa, h0


def evaluate(prepared, lam, attack_energy=ATTACK_ENERGY):
    eigval, eigvec, a_eig, kappa, h0 = prepared
    h_inc = kappa**2 * np.sum(
        a_eig**2 * eigval / (eigval + lam)**2, axis=1
    )
    health = h0 + float(np.mean(h_inc))

    g_eig = kappa[:, None] * a_eig * lam / (eigval + lam)
    g = np.einsum("ni,nji->nj", g_eig, eigvec)
    smat = g.T @ g / len(g)
    risk_fixed = health + attack_energy * float(np.linalg.eigvalsh(smat)[-1])
    risk_adaptive = health + attack_energy * float(np.trace(smat))
    return {
        "health": health,
        "risk_fixed": risk_fixed,
        "risk_adaptive": risk_adaptive,
        "S": smat,
        "health_increment": float(np.mean(h_inc)),
    }


def calibrate(prepared, target):
    """Bisection for the common health-budget regularizer."""
    lo, hi = 0.0, 1.0
    while evaluate(prepared, hi)["health"] > target:
        hi *= 2.0
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        if evaluate(prepared, mid)["health"] > target:
            lo = mid
        else:
            hi = mid
    return hi, evaluate(prepared, hi)


def summarize(rows):
    values = np.asarray(rows, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values)-1) * sd / math.sqrt(len(values)))
    return {"mean": mean, "sd_across_replicates": sd,
            "mean_95ci": [mean-half, mean+half]}


def run():
    h0 = PX / (1.0 + PX * float(np.sum(BX*BX*W)))
    target = (1.0 + HEALTH_BUDGET) * h0
    schemes = [
        ("iid", lambda n, s: sample_iid(n, s), 10_000, None),
        ("bscw_a025", lambda n, s: sample_bscw(n, s, 0.25), 61_000, 0.25),
        ("bscw_a030", lambda n, s: sample_bscw(n, s, 0.30), 62_000, 0.30),
        ("bscw_a035", lambda n, s: sample_bscw(n, s, 0.35), 20_000, 0.35),
        ("bscw_a040", lambda n, s: sample_bscw(n, s, 0.40), 30_000, 0.40),
        ("bscw_a045", lambda n, s: sample_bscw(n, s, 0.45), 40_000, 0.45),
        ("full_lc_tcw", lambda n, s: sample_full_lctcw(n, s), 50_000, None),
    ]

    result = {
        "settings": {
            "Px": PX, "health_budget_fraction": HEALTH_BUDGET,
            "local_block_attack_energy": ATTACK_ENERGY,
            "replicates": N_REPLICATES, "watermark_pairs_per_replicate": N_PER_REPLICATE,
            "sigma_diag": SIGMA.tolist(),
            "sigma_completion_assumption": "third variance fixed at 0.5; verify against frozen full model",
            "healthy_baseline": h0, "health_target": target,
        },
        "schemes": {},
    }

    for name, sampler, offset, alpha in schemes:
        lam_values, fixed_values, adaptive_values, health_values = [], [], [], []
        for seed in range(N_REPLICATES):
            beta = sampler(N_PER_REPLICATE, seed + offset)
            prepared = prepare(build_block(beta), PX)
            lam, metrics = calibrate(prepared, target)
            lam_values.append(lam)
            fixed_values.append(metrics["risk_fixed"])
            adaptive_values.append(metrics["risk_adaptive"])
            health_values.append(metrics["health"])

        if alpha is None:
            mi = 0.0 if name == "iid" else 5.6726771175687265
            tv = 0.0 if name == "iid" else 0.9948591554186776
            delta = None
        else:
            mi = -math.log(1.0 - 2.0*alpha)
            tv = 2.0*alpha
            delta = float(qchi(alpha))

        result["schemes"][name] = {
            "alpha": alpha, "mutual_information_nats": mi,
            "tv_from_iid_pair": tv, "coefficient_gap_delta": delta,
            "lambda": summarize(lam_values),
            "health": summarize(health_values),
            "risk_fixed": summarize(fixed_values),
            "risk_adaptive": summarize(adaptive_values),
        }

    # Separate information-matrix lower-tail diagnostic, not part of risk table.
    eig_diag = {}
    for name, sampler, seed, alpha in [
        ("iid", sample_iid, 100, None),
        ("bscw_a035", lambda n, s: sample_bscw(n, s, 0.35), 235, 0.35),
        ("bscw_a040", lambda n, s: sample_bscw(n, s, 0.40), 240, 0.40),
        ("full_lc_tcw", sample_full_lctcw, 300, None),
    ]:
        beta = sampler(100_000, seed)
        c = prepare(build_block(beta), PX)[0]
        eig_diag[name] = {
            "pairs": 100_000,
            "lambda_min_mean": float(np.mean(c[:, 0])),
            "lambda_min_p01": float(np.quantile(c[:, 0], 0.01)),
            "lambda_min_p05": float(np.quantile(c[:, 0], 0.05)),
            "lambda_min_median": float(np.median(c[:, 0])),
        }
    result["information_eigenvalue_diagnostic"] = eig_diag
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(out.name, "written")


if __name__ == "__main__":
    run()
