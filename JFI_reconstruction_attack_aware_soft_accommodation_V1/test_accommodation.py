"""Deterministic checks on the frozen references and already saved inputs."""
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import solve_triangular

from accommodation import OnlineEvidence, OnlineSoftCompensator, PreparedModel, Settings
from run_validation import DEFAULT_BASELINE, REFERENCE, load_json, open_baseline


class AccommodationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve().parent
        cls.record = load_json(here / "registration.json")
        cls.settings = Settings.from_registration(cls.record)
        cls.common = open_baseline(DEFAULT_BASELINE, cls.record)
        cls.models = {}
        cls.saved = np.load(DEFAULT_BASELINE / REFERENCE / "test_results.npz", allow_pickle=False)
        cls.addClassCleanup(cls.saved.close)
        for name in cls.record["models"]:
            model = cls.common.load_model(name, DEFAULT_BASELINE / REFERENCE / "inputs" / "five_node_aggregate_model.json")
            cls.models[name] = PreparedModel(model, cls.settings)

    def streams(self, name):
        y = self.saved[f"{name}__physical__y"]
        m = self.saved[f"{name}__Gaussian__M"]
        delta = self.saved[f"{name}__Gaussian__decoded_effect"]
        return y, m, delta

    def test_clean_codec_and_frozen_prediction_posterior(self):
        for name, p in self.models.items():
            with self.subTest(name=name):
                y, m, delta = self.streams(name)
                encoded, received = self.common.transmit_decode(p.model, y, m)
                for i, ids in enumerate(p.model.neighbors):
                    for j in ids:
                        np.testing.assert_allclose(received[:, i, j], y[:, j], atol=2e-12)
                actual = self.common.run_common(p.model, y, received)
                for key in ("prior", "posterior"):
                    np.testing.assert_allclose(actual[key], self.saved[f"{name}__Gaussian__{key}_clean"],
                                               rtol=1e-11, atol=1e-11)
                # Validate the archive's codec ordering using its recorded synthetic difference.
                target = p.settings.target
                saved_difference = self.saved[f"{name}__attack"]
                decoded_record = (m[:, target] @ (encoded[:, target] + saved_difference)[..., None])[..., 0]
                np.testing.assert_allclose(decoded_record - y[:, target], delta, rtol=1e-11, atol=1e-11)

    def test_reference_joint_covariance_against_error_recursion(self):
        for name, p in self.models.items():
            with self.subTest(name=name):
                s, model = p.settings, p.model
                k, g, a = model.K[s.receiver], model.G[s.receiver], model.A
                identity = np.eye(p.n)
                j = identity - k @ model.H[s.receiver] - g @ model.H[s.healthy]
                noise = k @ model.R[s.receiver] @ k.T + g @ model.R[s.healthy] @ g.T
                cross_state, error_cov = -identity, identity
                expected_post = []
                for t in range(s.horizon):
                    true_cross = cross_state @ model.H[s.target].T
                    np.testing.assert_allclose(p.reference_target_cross[t, t], true_cross, atol=4e-13)
                    np.testing.assert_allclose(p.reference_error_covariance[t], error_cov, atol=4e-13)
                    post_error = j @ error_cov @ j.T + noise
                    expected_post.append(np.trace(post_error))
                    cross_state = a @ j @ cross_state @ a.T - model.Q
                    error_cov = a @ post_error @ a.T + model.Q
                np.testing.assert_allclose(p.nominal_mse(target_accepted=False), expected_post, atol=4e-13)
                for t in (0, 7, 16, 31):
                    u = np.vstack(p.trusted_maps[:t + 1])
                    y = np.vstack(p.target_maps[:t + 1])
                    x = p.state_maps[t]
                    uu, yy, yu = u @ u.T, y @ y.T, y @ u.T
                    xy, xu = x @ y.T, x @ u.T
                    y_prior = yy - yu @ np.linalg.solve(uu, yu.T)
                    xy_prior = xy - xu @ np.linalg.solve(uu, yu.T)
                    x_prior = x @ x.T - xu @ np.linalg.solve(uu, xu.T)
                    prefix = p.prefixes[t]
                    np.testing.assert_allclose(prefix.target_cholesky @ prefix.target_cholesky.T, y_prior, atol=1e-10)
                    np.testing.assert_allclose(prefix.state_from_white @ prefix.target_cholesky.T, xy_prior, atol=1e-10)
                    np.testing.assert_allclose(prefix.conditional_state_cov,
                                               x_prior - xy_prior @ np.linalg.solve(y_prior, xy_prior.T), atol=1e-10)

    def test_likelihood_and_shift_moments_against_dense_formula(self):
        first = self.record["evaluation_onset"]
        for name, p in self.models.items():
            with self.subTest(name=name):
                clean, decoder, delta = self.streams(name)
                s = p.settings
                observer = OnlineEvidence(p, self.common.common_step)
                for k in range(s.horizon):
                    y = clean[k]
                    ev = observer.observe(y[s.receiver], y[s.healthy], y[s.target] + delta[k], decoder[k, s.target])
                    self.assertAlmostEqual(ev.weights.sum(), 1.0)
                    self.assertGreater(np.linalg.eigvalsh(ev.posterior_state_cov)[0], 0.0)
                    self.assertGreaterEqual(np.linalg.eigvalsh(ev.delta_cov)[0], -1e-11)
                    if k not in (first, first + 2, first + s.template_duration):
                        continue
                    stop = min(k + 1, first + s.template_duration)
                    d = np.zeros((k + 1, p.m, p.n))
                    d[first:stop] = decoder[first:stop, s.target] @ p.template[:stop - first]
                    prefix = p.prefixes[k]
                    b = d.reshape(-1, p.n) @ p.parameter_prior_cholesky
                    u = observer.trusted[:k + 1].reshape(-1)
                    z = observer.target[:k + 1].reshape(-1) - prefix.target_from_trusted @ u
                    covariance = prefix.target_cholesky @ prefix.target_cholesky.T
                    alternative = covariance + b @ b.T
                    log_bf = 0.5 * (np.linalg.slogdet(covariance)[1] - np.linalg.slogdet(alternative)[1]
                                    + z @ np.linalg.solve(covariance, z) - z @ np.linalg.solve(alternative, z))
                    bw = solve_triangular(prefix.target_cholesky, b, lower=True)
                    zw = solve_triangular(prefix.target_cholesky, z, lower=True)
                    normal = np.eye(p.n) + bw.T @ bw
                    predicted = b[-p.m:] @ np.linalg.solve(normal, bw.T @ zw)
                    variance = b[-p.m:] @ np.linalg.solve(normal, b[-p.m:].T)
                    np.testing.assert_allclose(ev.fits[first].log_bf, log_bf, atol=4e-10)
                    np.testing.assert_allclose(ev.fits[first].delta, predicted, atol=4e-11)
                    np.testing.assert_allclose(ev.fits[first].delta_cov, variance, atol=4e-11)

    def test_real_soft_and_full_paths_causality_guard_and_cross_term(self):
        onset = self.record["evaluation_onset"]
        for name, p in self.models.items():
            s, model = p.settings, p.model
            for full in (False, True):
                with self.subTest(name=name, full=full):
                    y, m, delta = self.streams(name)
                    truth = self.saved[f"{name}__physical__x"]
                    healthy = OnlineSoftCompensator(p, self.common.common_step, full)
                    attacked = OnlineSoftCompensator(p, self.common.common_step, full)
                    previous_bias = np.zeros(p.n)
                    for k in range(s.horizon):
                        local, neighbor = y[k, s.receiver], y[k, s.healthy]
                        sample, decoder = y[k, s.target], m[k, s.target]
                        normal = healthy.step(local, neighbor, sample, decoder)
                        fault = attacked.step(local, neighbor, sample + delta[k], decoder)
                        np.testing.assert_array_equal(fault.evidence.reference_posterior, normal.evidence.reference_posterior)
                        if k < onset:
                            np.testing.assert_array_equal(fault.posterior, normal.posterior)
                            np.testing.assert_array_equal(fault.evidence.weights, normal.evidence.weights)
                        for estimate in (normal, fault):
                            self.assertGreaterEqual(estimate.weight.alpha, 0.0)
                            self.assertLessEqual(estimate.weight.alpha, 1.0)
                            if not full:
                                budget = s.nominal_one_step_allowance * estimate.weight.risk_without_correction
                                self.assertLessEqual(estimate.weight.nominal_risk_increase, budget + 1e-12)
                                self.assertLessEqual(estimate.weight.robust_model_risk_increase, 1e-12)
                        diff = fault.posterior - normal.posterior
                        residual = delta[k] - fault.weight.alpha * fault.evidence.delta + normal.weight.alpha * normal.evidence.delta
                        expected = model.F(s.receiver) @ previous_bias + model.G[s.receiver] @ residual
                        np.testing.assert_allclose(diff, expected, atol=3e-12)
                        err, nominal = fault.posterior - truth[k], normal.posterior - truth[k]
                        self.assertAlmostEqual(err @ err, nominal @ nominal + diff @ diff + 2 * nominal @ diff, places=11)
                        previous_bias = diff


if __name__ == "__main__":
    unittest.main(verbosity=2)
