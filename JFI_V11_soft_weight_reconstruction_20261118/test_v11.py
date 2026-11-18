"""Deterministic checks of the scalar policy and imported nominal covariance."""
import unittest

import numpy as np

from analytical_weight import conditional_weight, nominal_limit, suffix_gramians
from weight_experiment import DEFAULT_BASELINE, REFERENCE, PreparedModel, setup


class WeightTests(unittest.TestCase):
    def test_suffix_propagates_posterior_correction(self):
        decay = np.array([[.83, .52], [0, .61]])
        size = 9
        grams = suffix_gramians(decay, size)
        for k, computed in enumerate(grams):
            expected = np.zeros((2, 2))
            for lag in range(size-k):
                power = np.linalg.matrix_power(decay, lag)
                expected += power.T @ power
            np.testing.assert_allclose(computed, expected, atol=4e-15)

    def test_exact_guard_boundary_and_stable_root(self):
        for curvature, linear, budget in [(1, -.25, .02), (1, .3, .02), (4, 1, 0),
                                          (1, -1e12, 1e-6), (1, 2, .02), (0, -1, .02)]:
            cap = nominal_limit(curvature, linear, budget)
            self.assertGreaterEqual(cap, 0)
            self.assertLessEqual(cap, 1)
            self.assertLessEqual(cap*(curvature*cap-2*linear), budget+2e-15)
            if cap < 1:
                self.assertAlmostEqual(cap*(curvature*cap-2*linear), budget, places=14)
                out = (cap+1)/2
                self.assertGreater(out*(curvature*out-2*linear), budget)

    def test_candidate_is_constrained_quadratic_minimum(self):
        direction = np.array([.4, -.2])
        preview = np.array([1., .32])
        model_null = preview - np.array([-.25, .18])
        budget = .016
        suffix = np.array([[1.8, .29], [.29, 1.4]])
        for target in [preview, preview-direction*.2, preview-1.25*direction, preview+direction]:
            weight = conditional_weight(preview, target, model_null, direction, suffix, budget)
            grid = np.linspace(0., weight.cap, 401)
            loss = np.sum((preview-target-grid[:, None]*direction)**2, axis=1)
            fit = np.sum((preview-target-weight.alpha*direction)**2)
            self.assertLessEqual(fit, loss.min()+1e-15)
            self.assertLessEqual(weight.tail_nominal_increment, budget+1e-15)
        zero = conditional_weight(preview, preview, model_null, np.zeros(2), suffix, budget)
        self.assertEqual(zero.alpha, 0.)
        self.assertEqual(zero.tail_nominal_increment, 0.)


class NominalMeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        common, settings, _, _ = setup(DEFAULT_BASELINE)
        cls.prepared = [PreparedModel(common.load_model(name, DEFAULT_BASELINE / REFERENCE /
                                                       "inputs" / "five_node_aggregate_model.json"), settings)
                        for name in ("stable", "unstable")]

    def test_nominal_conditioning_matches_full_gaussian_regression(self):
        for prepared in self.prepared:
            for k in (0, 7, 16, 31):
                state = prepared.state_maps[k]
                dim = state.shape[1]
                trusted = prepared.trusted_maps[:k+1].reshape(-1, dim)
                target = prepared.target_maps[:k+1].reshape(-1, dim)
                rows = np.vstack((trusted, target))
                direct = np.linalg.solve(rows @ rows.T, rows @ state.T).T
                prefix = prepared.prefixes[k]
                target_gain = np.linalg.solve(prefix.target_cholesky.T, prefix.state_from_white.T).T
                fitted = np.hstack((prefix.state_from_trusted-target_gain @ prefix.target_from_trusted, target_gain))
                np.testing.assert_allclose(fitted, direct, rtol=2e-8, atol=2e-10)
                residual = state-direct @ rows
                np.testing.assert_allclose(prefix.conditional_state_cov, residual @ residual.T, atol=1e-11)

    def test_nominal_mse_matches_unconditional_affine_error_covariance(self):
        for prepared in self.prepared:
            model, s = prepared.model, prepared.settings
            k_gain, g_gain = model.K[s.receiver], model.G[s.receiver]
            f = model.F(s.receiver)
            estimate = np.zeros_like(prepared.state_maps[0])
            mse = []
            for k in range(s.horizon):
                obs = prepared.trusted_maps[k].reshape(2, prepared.m, -1)
                estimate = f @ estimate + k_gain @ obs[0] + g_gain @ (obs[1]+prepared.target_maps[k])
                mse.append(np.sum((estimate-prepared.state_maps[k])**2))
            np.testing.assert_allclose(mse, prepared.nominal_mse(), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
