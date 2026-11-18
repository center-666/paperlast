"""Deterministic checks using frozen healthy records."""
import unittest

import numpy as np

from budget_policy import CarryBudgetPolicy
from shared import DEFAULT_BASELINE, REFERENCE, V11, load_json, load_prepared, setup
from nominal_audit import bounded_mean_upper, null_batch, rational_envelope, regression_cache
from receiver_experiment import METHODS, exhaust_ledger, stack
from run_validation import gain_on_nominal_grid
from validate_v11 import paired_packets, replay_bank
from weight_experiment import make_parity


class BudgetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common, cls.settings, cls.spec, _ = setup(DEFAULT_BASELINE)
        cls.registered = load_json(V11 / "phase2_registration.json")

    def test_frozen_health_law_and_full_receiver_equality(self):
        for name in ("unstable", "stable"):
            p = load_prepared(DEFAULT_BASELINE, name, self.common, self.settings)
            path = DEFAULT_BASELINE / REFERENCE / "inputs/five_node_aggregate_model.json"
            stage, base = rational_envelope(path, name, self.spec)
            np.testing.assert_allclose(stage, p.nominal_mse(), atol=1e-12)
            np.testing.assert_allclose(base, [np.trace(a.conditional_state_cov) for a in p.prefixes], atol=1e-12)
            ordinary, bank = replay_bank(DEFAULT_BASELINE, name, self.registered)
            cases, _ = paired_packets(self.spec, self.registered, self.common, p.model, ordinary, bank)
            episodes = cases["healthy"][:3]
            prior = np.stack([np.c_[s.local, s.healthy] for s in episodes])
            target = np.stack([s.clean_target for s in episodes])
            decoder = np.stack([s.decoders for s in episodes])
            posts, scores = null_batch(p, prior, target, decoder, regression_cache(p))
            scale = gain_on_nominal_grid(p, self.spec)
            parity = make_parity(self.common, p, self.spec)
            replay = stack(episodes, [7]*len(episodes), p, self.common, self.spec, scale, parity)
            np.testing.assert_allclose(posts, replay["post"][:, :4], atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(scores+base, replay["conditional_null_mse"][:, :4], atol=1e-11)
            proof = exhaust_ledger(replay, p)
            self.assertLessEqual(proof["maximum_path_telescope_error"], 1e-10)
            j = METHODS.index("V1.2a")
            self.assertTrue((replay["balance"][:, j] >= 0.).all())

    def test_receive_timestamp_guard(self):
        p = load_prepared(DEFAULT_BASELINE, "stable", self.common, self.settings)
        policy = CarryBudgetPolicy(p)
        with self.assertRaises(ValueError):
            policy.step(1, None, None)
        self.assertEqual(policy.credit, 0.)

    def test_bounded_mean_inversion(self):
        clip, alpha = 2.65, .05/6
        data = np.array([0., .04, .12, .1, 12.])
        upper = bounded_mean_upper(data, clip, alpha)
        t = .95/clip
        e_factor = (1-t*np.minimum(data, clip))/(1-t*upper)
        self.assertAlmostEqual(np.log(e_factor).sum(), -np.log(alpha), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
