"""Gaussian quadrature integrand for the saved A0 receiver experiment."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, logsumexp, ndtri
from scipy.stats import qmc

from exact_bounds import START, LAST


AGES = LAST - START + 1
HORIZON = 32
BF_THRESHOLD = 20
DANGER_SQUARED = 0.1
LOG_THRESHOLD = np.log(BF_THRESHOLD)
SCRAMBLES = (8539, 12149, 19381, 27397)
POWERS = (9, 10)


@dataclass(frozen=True)
class Prefix:
    whitener: np.ndarray
    white_coordinates: np.ndarray
    state_from_white: np.ndarray


@dataclass(frozen=True)
class WorkPosterior:
    log_bf: np.ndarray
    log_bf_by_onset: np.ndarray
    active_probability: np.ndarray
    correction: np.ndarray


def work_posterior(design, prefix, white):
    size, n, _, m, rank = design.shape
    x = prefix.whitener @ design.reshape(size, n, n*m, rank)
    xt = x.swapaxes(-2, -1)
    precision = np.eye(rank) + xt @ x
    information = xt @ white[:, None, :, None]
    coef = np.linalg.solve(precision, information)
    profile_log_bf = 0.5 * (
        np.sum((information*coef)[..., 0], axis=-1) - np.linalg.slogdet(precision)[1]
    )
    partial = logsumexp(profile_log_bf, axis=1)
    future = HORIZON - n
    # Before and including the third packet, every begun working template is active.
    log_normalizer = np.logaddexp(partial, np.log(HORIZON + future))
    weights = np.exp(profile_log_bf - log_normalizer[:, None])
    log_bf = np.logaddexp(partial, np.log(future)) - np.log(HORIZON)
    correction = np.sum(weights[..., None] * (design[:, :, -1] @ coef)[..., 0], axis=1)
    return WorkPosterior(log_bf, profile_log_bf, weights.sum(axis=1), correction)


def ball(x, radius):
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return x * (radius / np.maximum(radius, length))


def prefix_events(log_bf, displacement):
    surviving = np.maximum.accumulate(log_bf,axis=1)[:, START:LAST+1] < LOG_THRESHOLD
    danger = np.sum(displacement**2, axis=-1) >= DANGER_SQUARED
    event = danger & surviving[:, None, :]
    return event, np.logical_or.accumulate(event, axis=-1)


class JointIntegral:
    """Evaluate the same latent variables in the healthy and fixed-A0 integrands."""

    def __init__(self, p, saved_profile, radii):
        self.p = p
        self.radii = tuple(radii.items())
        self.names = ("V0", "V1a", *radii)
        self.fixed_profile = saved_profile[:AGES].copy()
        self.design = p.template @ p.parameter_prior_cholesky
        self.F = p.model.F(p.settings.receiver)
        self.G = p.model.G[p.settings.receiver]
        trusted = (p.settings.receiver, p.settings.healthy)
        self.score = np.hstack([p.model.H[j].T @ np.linalg.inv(p.model.R[j]) for j in trusted])
        d = p.target_maps.shape[-1]
        target = p.target_maps[:LAST+1].reshape(-1, d)
        scores = (self.score @ p.trusted_maps[:LAST+1]).reshape(-1, d)
        joint = np.vstack((scores, target))
        self.joint_cholesky = np.linalg.cholesky(joint @ joint.T)
        basis = solve_triangular(self.joint_cholesky, joint, lower=True)
        self.prefixes = []
        for k in range(LAST+1):
            prefix = p.prefixes[k]
            t = p.trusted_maps[:k+1].reshape(-1, d)
            y = p.target_maps[:k+1].reshape(-1, d)
            transform = solve_triangular(
                prefix.target_cholesky, np.eye((k+1)*p.m), lower=True
            )
            white = transform @ (y-prefix.target_from_trusted @ t)
            coordinates = white @ basis.T
            np.testing.assert_allclose(coordinates @ basis, white, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(
                coordinates @ coordinates.T, np.eye((k+1)*p.m), atol=2e-12, rtol=2e-12
            )
            self.prefixes.append(Prefix(transform, coordinates, prefix.state_from_white))

    @property
    def dimension(self):
        return self.joint_cholesky.shape[0] + (LAST+1)*self.p.m

    def archive_coordinates(self, trusted, target):
        count = len(target)
        scores = (trusted[:, :LAST+1] @ self.score.T).reshape(count, -1)
        y = target[:, :LAST+1].reshape(count, -1)
        joint = np.concatenate((scores, y), axis=1)
        return solve_triangular(self.joint_cholesky, joint.T, lower=True).T

    def evaluate(self, standard, key):
        size = len(standard)
        m, rank = self.p.m, self.p.n
        difference = np.zeros((size,LAST+1,m))
        # An archived deterministic likelihood displacement, with the receiver's decoder law.
        difference[:, START:] = (key[:, START:] @ self.fixed_profile[..., None])[..., 0]
        offsets = np.zeros((size,len(self.names),AGES,rank))
        alarm = np.empty((size,LAST+1))
        prior_scores = np.empty((size,AGES,4))
        v0 = np.zeros((size,rank))
        v1a = np.zeros_like(v0)
        for k, prefix in enumerate(self.prefixes):
            n = k+1
            clean_white = standard @ prefix.white_coordinates.T
            design = np.zeros((size,n,n,m,rank))
            for start in range(n):
                design[:, start, start:] = key[:, start:n] @ self.design[:n-start]
            if k < START:
                alarm[:, k] = work_posterior(design, prefix, clean_white).log_bf
                continue
            j = k-START
            shift = difference[:, :n].reshape(size, -1) @ prefix.whitener.T
            observed_white = clean_white + shift
            reference = work_posterior(design, prefix, clean_white)
            actual = work_posterior(design, prefix, observed_white)
            alarm[:, k] = actual.log_bf
            v0 = v0 @ self.F.T + difference[:, k] @ self.G.T
            v1a = v1a @ self.F.T + (
                difference[:, k] - actual.correction + reference.correction
            ) @ self.G.T
            offsets[:, 0, j] = v0
            offsets[:, 1, j] = v1a
            clean_gap = clean_white @ prefix.state_from_white.T
            observed_gap = observed_white @ prefix.state_from_white.T
            for i, (_, radius) in enumerate(self.radii, start=2):
                offsets[:, i, j] = ball(observed_gap, radius)-ball(clean_gap, radius)
            # The last three scores are offline likelihood diagnostics under equal prior odds.
            prior_scores[:, j, 0] = actual.active_probability
            prior_scores[:, j, 1] = expit(logsumexp(actual.log_bf_by_onset, axis=1)-np.log(n))
            prior_scores[:, j, 2] = expit(actual.log_bf_by_onset[:, START])
            prior_scores[:, j, 3] = expit(np.sum(shift*(observed_white-shift/2), axis=1))
        return alarm, offsets, prior_scores


def quadrature(integrand):
    summaries = {"nodes_power_"+str(power): [] for power in POWERS}
    for seed in SCRAMBLES:
        cube = qmc.Sobol(d=integrand.dimension,scramble=True,seed=seed).random_base2(POWERS[-1])
        normals = ndtri(cube)
        d = integrand.joint_cholesky.shape[0]
        beta = normals[:, d:].reshape(len(cube), LAST+1, integrand.p.m) / np.sqrt(2.)
        key = 0.5*np.eye(integrand.p.m) + beta[..., None] * beta[..., None, :]
        alarm, offsets, scores = integrand.evaluate(normals[:, :d], key)
        event, cumulative = prefix_events(alarm,offsets)
        for power in POWERS:
            keep = slice(0, 2**power)
            marginal = event[keep, 1]
            rows = [[float(scores[keep, j, h][marginal[:, j]].mean())
                     if marginal[:, j].any() else None for h in range(4)] for j in range(AGES)]
            summaries["nodes_power_"+str(power)].append({
                "seed": seed,
                "risk_by_method_by_age": cumulative[keep].mean(axis=0).tolist(),
                "v1a_score_mean_at_strict_event": rows,
                "v1a_strict_event_counts_at_age": marginal.sum(axis=0).tolist(),
            })
    return {
        "methods": integrand.names,
        "nodes_per_replication": [2**i for i in POWERS],
        "dimension": integrand.dimension,
        "checkpoint_dimensions_including_watermarks": [6*(k+1) for k in range(START,LAST+1)],
        "score_columns": ["frozen_active_probability", "equal_odds_mixed_start_work",
                          "equal_odds_fixed_start_work", "equal_odds_fixed_a0"],
        "replications": summaries,
        "classification": "Randomized Sobol quadrature diagnostics, no interval enclosure or ranking certificate",
    }
