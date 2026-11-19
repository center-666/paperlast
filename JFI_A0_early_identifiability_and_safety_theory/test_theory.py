"""Checks for Gaussian prefix identities, model alignment and paired propagation."""

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from early_experiment import gaussian_errors, mean_distance_squared, mixture_information, mixture_risk
from run_theory import evaluate


class GaussianIdentityTests(unittest.TestCase):
    def test_even_rule_error_agrees_with_direct_density_overlap(self):
        d = 1.5
        signed, even = gaussian_errors(np.array([0., d]))
        integrated = quad(lambda x: 0.5*min(
            norm.pdf(x), (norm.pdf(x-d)+norm.pdf(x+d))/2
        ), -12, 12, epsabs=1e-9)[0]
        self.assertAlmostEqual(signed[0], .5)
        self.assertAlmostEqual(even[0], .5)
        self.assertGreater(even[1], signed[1])
        self.assertAlmostEqual(even[1], integrated, places=8)

    def test_identical_mean_components_have_zero_label_loss(self):
        means = np.array([[[1., 0.], [1., 0.]]])
        error = mixture_risk(means, 32)[0]
        kl, loss, labelled = mixture_information(means, 32)
        self.assertAlmostEqual(error, norm.cdf(-.5), places=12)
        self.assertAlmostEqual(kl[0], .5, places=12)
        self.assertAlmostEqual(loss[0], 0, places=12)
        self.assertAlmostEqual(labelled[0], .5, places=12)

    def test_watermark_second_moment_against_isotropic_formula(self):
        mean, covariance = mean_distance_squared(np.eye(2), np.array([[1., 0.]]), .5, .5)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(covariance, .75)

    def test_distinct_mixture_means_strictly_lose_label_information(self):
        shifts = np.array([[[1., 0.], [0., 1.5]]])
        errors, _ = gaussian_errors(np.linalg.norm(shifts, axis=2))
        self.assertGreater(mixture_risk(shifts, 128)[0], errors.mean())
        kl, loss, labelled = mixture_information(shifts, 128)
        self.assertGreater(loss[0], 0)
        self.assertLess(loss[0], np.log(2))
        self.assertAlmostEqual(kl[0]+loss[0], labelled[0])


class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = evaluate()

    def test_saved_models_match_energy_and_renormalize(self):
        self.assertEqual(self.results["new_physical_paths"], 0)
        for model in self.results["models"].values():
            audit = model["model_consistency"]
            self.assertAlmostEqual(audit["coded_energy_verified"], 10)
            self.assertAlmostEqual(audit["working_gaussian_probability_full_template_energy_exceeds_budget"],
                                   np.exp(-1))
            self.assertLess(model["posterior_audit"]["max_partition_residual"], 1e-13)
            self.assertLess(model["frozen_replay_max_absolute_error"], 1e-10)

    def test_early_error_gaps_and_numeric_quadrature(self):
        for model in self.results["models"].values():
            for phase in model["first_three_prefixes"].values():
                fixed = phase["fixed_start_7_equal_prior"]
                unknown = phase["active_start_3_or_7_equal_prior"]
                self.assertEqual(phase["conditioned_decoder_records"], 128)
                self.assertGreater(fixed["even_rule_penalty"], 0)
                self.assertGreater(unknown["error_paid_for_hiding_label"], 0)
                self.assertLess(unknown["risk_64_128_max_abs"], 1e-5)
                self.assertLess(unknown["kl_64_128_max_abs"], 1e-5)
                self.assertAlmostEqual(unknown["kl_mixture_average"] +
                                       unknown["conditional_label_information_nats_average"],
                                       unknown["kl_labelled_average"])

    def test_actual_mse_expansion_and_conditional_risk_scope(self):
        for model in self.results["models"].values():
            bridge = model["common_g_bridge"]
            self.assertLess(bridge["max_recurrence_error"], 1e-10)
            self.assertLess(bridge["max_state_mse_expansion_error"], 1e-12)
            for r in bridge["first_three"].values():
                self.assertAlmostEqual(r["nominal_state_mse"] + r["paired_state_displacement_mean_squared"]
                                       + r["state_error_cross_term"], r["actual_state_mse"])
                self.assertGreaterEqual(r["weighted_L2_propagation_upper"] + 1e-12,
                                        r["paired_state_displacement_mean_squared"])
                self.assertGreaterEqual(r["empirical_alarm_gated_union_upper_through_k"],
                                        r["empirical_strict_pre_alarm_count_through_k"]/12)

    def test_fixed_v0_lower_bound_uses_only_first_crossing_decoder_strata(self):
        for model in self.results["models"].values():
            lower = 0
            for row in model["first_three_prefixes"].values():
                bound = row["v0_before_bf_declaration_within_prefix"]
                value = bound["empirical_decoder_law_pre_alarm_lower_bound"]
                self.assertGreaterEqual(value, lower)
                self.assertLessEqual(value, bound["dangerous_decoder_records_by_this_prefix"]/128)
                lower = value

    def test_nominal_only_match_is_replayed_on_twelve_records_each(self):
        for model in self.results["models"].values():
            matched, replay = model["nominal_matching"], model["matched_replay"]
            self.assertAlmostEqual(matched["v1a_nominal_target_estimate"],
                                   matched["achieved_gaussian_formula_cost"], places=12)
            lo, hi = matched["radius_range_from_target_approximate_95_interval"]
            self.assertLess(lo, matched["matched_radius_for_point_estimate"])
            self.assertGreater(hi, matched["matched_radius_for_point_estimate"])
            self.assertEqual(len(matched["frozen_points"]), 4)
            pair = replay["matched_minus_v1a_active_mse_paired_t95_exploratory"]
            self.assertEqual(pair["n"], 12)
            self.assertLess(pair["interval"][0], 0)
            self.assertGreater(pair["interval"][1], 0)


if __name__ == "__main__":
    unittest.main()
