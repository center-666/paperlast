"""Sender-shared 150-step watermark keys for frozen 50-node common-G messages."""
from __future__ import annotations

import math
from functools import lru_cache
from scipy.interpolate import CubicHermiteSpline
from scipy.special import i0

import numpy as np

from watson_crn import CommonDraws, beta_from_common, make_common_draws, von_mises_icdf_signed

T = 150
SENDERS = 50
THETA_IID = 0.0
THETA_NEGATIVE = -1.3152369768257723


@lru_cache(None)
def inverse_table(theta):
    u = np.linspace(0., 1., 4097)
    phase = np.r_[von_mises_icdf_signed(u[:-1], theta), 2*np.pi]
    slope = 2*np.pi*i0(theta/2)*np.exp(-theta/2*np.cos(phase))
    return CubicHermiteSpline(u, phase, slope)


def phase_for(u, theta):
    return inverse_table(theta)(u) if theta else (2*np.pi)*u


def beta_chain(draws, phase):
    r = draws.radial_chi2_half
    angle = np.empty_like(r)
    angle[:, 0] = draws.initial_angle
    for t in range(1, r.shape[1]):
        angle[:, t] = (angle[:, t-1] + .5*phase[:, t-1] + np.pi*draws.branch[:, t-1]) % (2*np.pi)
    return np.sqrt(r)[..., None]*np.stack((np.cos(angle),np.sin(angle)), axis=-1)


def beta_radial_binned(draws: CommonDraws, regular_theta: float) -> np.ndarray:
    """Use 2t iff both neighboring exponential radii exceed ln(2)."""
    radii = draws.radial_chi2_half
    if radii.ndim != 3 or radii.shape[1] != T:
        raise ValueError("Expected complete [path,150,sender] radial keys")
    strong = (radii[:, :-1] > math.log(2.0)) & (radii[:, 1:] > math.log(2.0))
    return beta_chain(draws, np.where(strong, phase_for(draws.quantile_u, 2*regular_theta),
                                      phase_for(draws.quantile_u, regular_theta)))


def beta_for_scheme(draws: CommonDraws, scheme: str) -> np.ndarray:
    if scheme == "iid":
        theta = THETA_IID
        return beta_chain(draws, phase_for(draws.quantile_u, theta))
    if scheme == "negative_fixed":
        return beta_chain(draws, phase_for(draws.quantile_u, THETA_NEGATIVE))
    if scheme == "negative_radial_bin":
        return beta_radial_binned(draws, -1.0149016255359247)
    if scheme == "negative_random_bin":
        return beta_random_binned(draws, -1.0149016255359247)
    raise ValueError(f"Unknown watermark schedule {scheme}")


def beta_random_binned(draws: CommonDraws, regular_theta: float) -> np.ndarray:
    """Independent, sender-shared Bernoulli labels supplied by their own RNG."""
    return beta_chain(draws, np.where(draws.radial_bin_labels,
                      phase_for(draws.quantile_u, 2*regular_theta),
                      phase_for(draws.quantile_u, regular_theta)))


def make_key_draws(count: int, radial_seed: int, initial_seed: int,
                   transition_seed: int, random_bin_seed: int | None = None) -> CommonDraws:
    draws = make_common_draws(
        count, T, SENDERS, radial_seed, initial_seed, transition_seed
    )
    if random_bin_seed is not None:
        rng = np.random.default_rng(random_bin_seed)
        draws.radial_bin_labels = rng.random(draws.quantile_u.shape) < 0.25
    return draws


def matrices_from_beta(beta: np.ndarray) -> np.ndarray:
    if beta.ndim != 4 or beta.shape[1:] != (T, SENDERS, 2):
        raise ValueError("Expected beta[path,150,50,2] for sender-shared keys")
    return 0.5 * np.eye(2) + np.einsum("btni,btnj->btnij", beta, beta)


def clean_encode_decode(model, physical, matrices):
    encoded = np.linalg.solve(matrices, physical.y[..., None])[..., 0]
    decoded = np.einsum("btnij,btnj->btni", matrices, encoded)
    return decoded


def attack_delta_for_schedule(model, matrices, raw_attack, onset: int) -> np.ndarray:
    if onset < 0 or onset + 60 > T:
        raise ValueError("attack window must fit the 150-step horizon")
    delta = np.zeros((len(matrices), T, 6, 2), dtype=float)
    senders = model.links[:, 1].astype(int)
    delta[:, onset:onset + 60] = np.einsum(
        "bteij,tej->btei",
        matrices[:, onset:onset + 60, senders],
        raw_attack[50:110],
    )
    return delta
