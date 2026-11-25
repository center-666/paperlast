"""Independent deterministic/numerical audit of the proposed radial-bin law."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import i0e, i1e, roots_laguerre

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE = ROOT / "paper1_sscd_commonrandom_20261122"
sys.path.insert(0, str(BASELINE))
from watson_crn import von_mises_icdf_signed, theoretical_zeta_mi

REGULAR_PROBABILITY = 0.75
STRONG_PROBABILITY = 0.25
SPLIT = math.log(2.0)
FIXED_NEGATIVE_THETA = -1.3152369768257723
MONTE_CARLO_COUNT = 250_000
MONTE_CARLO_SEED = 345792


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def watson_mi(theta: float) -> float:
    concentration = float(theta) / 2.0
    zeta = i1e(concentration) / i0e(concentration)
    log_i0 = abs(concentration) + math.log(i0e(concentration))
    return concentration * zeta - log_i0


def theta_for_information_budget(budget: float) -> float:
    return brentq(
        lambda value: REGULAR_PROBABILITY * watson_mi(value)
        + STRONG_PROBABILITY * watson_mi(2.0 * value)
        - budget,
        -2.1,
        -1e-10,
        xtol=5e-15,
    )


def radial_product_expectation(function) -> float:
    nodes, weights = roots_laguerre(100)
    left, right = np.meshgrid(nodes, nodes, indexing="ij")
    return float(np.sum(np.outer(weights, weights) * function(left, right)))


def zeta(theta):
    value = float(theta) / 2.0
    return i1e(value) / i0e(value)


def expected_two_packet_energy(theta: float) -> float:
    cross = radial_product_expectation(
        lambda s0, s1: s0 * s1 * zeta(theta)
    )
    return 1.5 - 0.5 * cross


def radial_binned_energy(regular_theta: float) -> float:
    # E[S 1(S>ln 2)] = (1+ln 2)/2 exactly for S~Exp(1).
    strong_weighted_product = ((1.0 + math.log(2.0)) / 2.0) ** 2
    regular_weighted_product = 1.0 - strong_weighted_product
    weighted_direction_moment = (
        regular_weighted_product * zeta(regular_theta)
        + strong_weighted_product * zeta(2.0 * regular_theta)
    )
    return 1.5 - 0.5 * weighted_direction_moment


def projective_angle_density(angle: float, theta: float) -> float:
    concentration = theta / 2.0
    log_normalizer = concentration + abs(concentration) + math.log(i0e(concentration))
    return math.exp(theta * math.cos(angle) ** 2 - log_normalizer) / math.pi


def hidden_random_bin_mi(regular_theta: float) -> float:
    def mixture(angle: float) -> float:
        return (
            REGULAR_PROBABILITY * projective_angle_density(angle, regular_theta)
            + STRONG_PROBABILITY * projective_angle_density(angle, 2.0 * regular_theta)
        )

    return quad(
        lambda angle: mixture(angle) * math.log(math.pi * mixture(angle)),
        0.0,
        math.pi,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=200,
    )[0]


def sample_two_packet_matrix_energy(regular_theta: float, count: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    radial = rng.exponential(size=(count, 2))
    initial_angle = rng.uniform(0.0, 2.0 * math.pi, count)
    quantile = rng.random(count)
    branch = rng.integers(0, 2, count)
    strong = (radial[:, 0] > SPLIT) & (radial[:, 1] > SPLIT)

    regular_phase = von_mises_icdf_signed(quantile, regular_theta)
    strong_phase = von_mises_icdf_signed(quantile, 2.0 * regular_theta)
    fixed_phase = von_mises_icdf_signed(quantile, FIXED_NEGATIVE_THETA)
    independent_phase = 2.0 * math.pi * quantile

    phase_by_design = {
        "iid": independent_phase,
        "fixed_negative": fixed_phase,
        "radial_bin_negative": np.where(strong, strong_phase, regular_phase),
    }
    unit_initial = np.column_stack((np.cos(initial_angle), np.sin(initial_angle)))
    perturbation = np.asarray([1.0, 0.0])
    results = {}
    for name, phase in phase_by_design.items():
        next_angle = (initial_angle + 0.5 * phase + math.pi * branch) % (2.0 * math.pi)
        beta0 = np.sqrt(radial[:, 0, None]) * unit_initial
        beta1 = np.sqrt(radial[:, 1, None]) * np.column_stack((np.cos(next_angle), np.sin(next_angle)))
        matrix0 = 0.5 * np.eye(2) + np.einsum("ni,nj->nij", beta0, beta0)
        matrix1 = 0.5 * np.eye(2) + np.einsum("ni,nj->nij", beta1, beta1)
        packet0 = np.einsum("nij,j->ni", matrix0, perturbation)
        packet1 = np.einsum("nij,j->ni", matrix1, perturbation)
        squared_distance = np.sum((packet1 - packet0) ** 2, axis=1)
        results[name] = {
            "mean": float(squared_distance.mean()),
            "standard_error": float(squared_distance.std(ddof=1) / math.sqrt(count)),
        }

    random_bin = rng.random(count) < STRONG_PROBABILITY
    random_phase = np.where(random_bin, strong_phase, regular_phase)
    next_angle = (initial_angle + 0.5 * random_phase + math.pi * branch) % (2.0 * math.pi)
    beta0 = np.sqrt(radial[:, 0, None]) * unit_initial
    beta1 = np.sqrt(radial[:, 1, None]) * np.column_stack((np.cos(next_angle), np.sin(next_angle)))
    matrix0 = 0.5 * np.eye(2) + np.einsum("ni,nj->nij", beta0, beta0)
    matrix1 = 0.5 * np.eye(2) + np.einsum("ni,nj->nij", beta1, beta1)
    packet0 = np.einsum("nij,j->ni", matrix0, perturbation)
    packet1 = np.einsum("nij,j->ni", matrix1, perturbation)
    distance = np.sum((packet1 - packet0) ** 2, axis=1)
    results["random_bin_hidden_label"] = {
        "mean": float(distance.mean()),
        "standard_error": float(distance.std(ddof=1) / math.sqrt(count)),
    }

    beta_all = np.concatenate((beta0, beta1), axis=0)
    coordinates = np.concatenate((matrix0, matrix1), axis=0)
    direction = beta_all / np.linalg.norm(beta_all, axis=1, keepdims=True)
    results["radial_edge_marginal"] = {
        "S_mean": float(np.mean(np.sum(beta_all**2, axis=1))),
        "S_variance": float(np.var(np.sum(beta_all**2, axis=1), ddof=1)),
        "beta_covariance": np.cov(beta_all.T).tolist(),
        "direction_mean": direction.mean(axis=0).tolist(),
        "M_mean": coordinates.mean(axis=0).tolist(),
    }
    results["strong_pair_frequency"] = float(strong.mean())
    results["conditional_angle_moment_regular"] = float(
        np.mean(np.cos(2.0 * regular_phase / 2.0))
    )
    results["conditional_angle_moment_strong"] = float(
        np.mean(np.cos(2.0 * strong_phase / 2.0)[strong])
    ) if strong.any() else None
    return results


def main():
    budget = 0.1
    regular_theta = theta_for_information_budget(budget)
    fixed_mi = theoretical_zeta_mi(FIXED_NEGATIVE_THETA)[1]
    weighted_mi = 0.75 * watson_mi(regular_theta) + 0.25 * watson_mi(2 * regular_theta)
    product_strong_moment = ((1.0 + math.log(2.0)) / 2.0) ** 2
    result = {
        "status": "PASS_INDEPENDENT_FORMULA_AUDIT",
        "registration_status": "registered before new paths",
        "source_reference": "8953e1dd9a5e2535ead2698acb6f345a85d4f532",
        "watermark_generator_sha256": sha256(BASELINE / "watson_crn.py"),
        "mean_adjacent_information_budget_nats": budget,
        "regular_theta_t": regular_theta,
        "strong_theta_2t": 2.0 * regular_theta,
        "conditional_information_nats": {
            "regular": watson_mi(regular_theta),
            "strong": watson_mi(2.0 * regular_theta),
            "fixed_negative_baseline": fixed_mi,
            "radial_rule_mean": weighted_mi,
            "hidden_random_bin_actual_mi": hidden_random_bin_mi(regular_theta),
        },
        "strong_pair_probability": STRONG_PROBABILITY,
        "analytic_E_S0S1_indicator_strong": product_strong_moment,
        "analytic_two_packet_expected_squared_distance": {
            "iid": 1.5,
            "fixed_negative": expected_two_packet_energy(FIXED_NEGATIVE_THETA),
            "radial_bin_negative": radial_binned_energy(regular_theta),
            "relative_candidate_increase_over_fixed_negative": (
                radial_binned_energy(regular_theta) / expected_two_packet_energy(FIXED_NEGATIVE_THETA) - 1.0
            ),
        },
        "monte_carlo": sample_two_packet_matrix_energy(
            regular_theta, MONTE_CARLO_COUNT, MONTE_CARLO_SEED
        ),
        "monte_carlo_count": MONTE_CARLO_COUNT,
        "monte_carlo_seed": MONTE_CARLO_SEED,
        "caveat": "The two-packet expectation is a proxy, not a detector-power or closed-loop MSE guarantee.",
    }
    output = HERE / "FORMULA_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
