"""Independent checks against the frozen observer and affine recursion."""
import unittest
from dataclasses import replace

import numpy as np

from inputs import DEFAULT_BASELINE, legacy_cases, load_prepared, setup
from batched_observer import BatchedObserver
from replay import Observations, play, summarize
from risk_policy import OnlineRiskReceiver, circle_grid, make_risk, radial_tail
from weight_experiment import OnlineEvidence


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common, cls.settings, cls.spec, _ = setup(DEFAULT_BASELINE)
        cls.p = load_prepared(DEFAULT_BASELINE, "unstable", cls.common, cls.settings)
        cls.examples = legacy_cases(DEFAULT_BASELINE, cls.p, cls.spec, cls.common)[0]

    def test_isotropic_and_degenerate_angular_integral(self):
        theta = circle_grid(32)
        v = np.array([.1, .45, .7])
        got = radial_tail(np.zeros((3,2)), v[:,None,None]*theta[None], .1)
        np.testing.assert_allclose(got, np.exp(-.1/(2*v*v)), atol=2e-15)
        got = radial_tail(np.array([[0.,0.],[1.,0.]]), np.zeros((2,2,32)), .1)
        np.testing.assert_array_equal(got, [0.,1.])

    def test_frozen_observer_moments(self):
        p = self.p
        subset = self.examples["A1_C_mismatch"].take(slice(0,1))
        reference = OnlineEvidence(p,self.common.common_step)
        fused = BatchedObserver(p)
        for k in range(self.settings.horizon):
            ev = reference.observe(subset.trusted[0,k,:p.m],subset.trusted[0,k,p.m:],
                                   subset.target[0,k],subset.decoder[0,k])
            frame = fused.at(subset,k,risk=True)
            np.testing.assert_allclose(frame.delta[0], ev.delta, atol=1e-11)
            np.testing.assert_allclose(frame.mean_working[0], ev.mixture_state_mean, atol=2e-11)
            np.testing.assert_allclose(frame.mean_null[0], ev.nominal_state_mean, atol=2e-11)
            np.testing.assert_allclose(frame.posterior_weights[0], ev.weights[1:k+2], atol=1e-12)
            np.testing.assert_allclose(frame.log_bf[0], ev.log_event_bf, atol=1e-10)

    def test_jacobian_small_observation_variation(self):
        p = self.p
        src = self.examples["healthy"].take(slice(0,1))
        obs = BatchedObserver(p)
        frame = obs.at(src, 7, risk=True)
        vec = np.array([.13,-.22])
        direction = np.zeros_like(src.target)
        direction[:,7] = frame.transfer[:,7] @ vec
        plus = obs.at(replace(src,target=src.target+1e-5*direction),7).delta
        minus = obs.at(replace(src,target=src.target-1e-5*direction),7).delta
        np.testing.assert_allclose((plus-minus)/(2e-5),
                                   frame.shift_sensitivity[:,7] @ vec, atol=2e-9)

    def test_online_matches_paired_replay_with_no_truth_leak(self):
        p = self.p
        test = self.examples["A0"].take(slice(0,3))
        clean = self.examples["healthy"].take(slice(0,3))
        idx = ("V1a","V1.3")
        factory = make_risk({"V1.3":1.},.1,16)
        expected = play(p,test,idx,factory)
        compare = play(p,clean,idx,factory)
        summarize(p,expected,compare,test,idx,.1)
        obs = Observations(test.trusted,test.target,test.decoder)
        rx = OnlineRiskReceiver(p,self.common.common_step,1.)
        for k in range(self.settings.horizon):
            got = rx.step(obs.trusted[0,k,:p.m],obs.trusted[0,k,p.m:],obs.target[0,k],obs.decoder[0,k])
            np.testing.assert_allclose(got.estimate,expected["post"][0,1,k],atol=2e-11)
            self.assertEqual(got.weight,expected["alpha"][0,1,k])
            if got.risk_score is not None:
                self.assertAlmostEqual(got.risk_score,expected["predicted"][0,1,k],places=11)
        changed = test.target.copy()
        changed[:,16:] += 1.
        later = play(p,replace(test,state=np.zeros_like(test.state),target=changed),idx,factory)
        np.testing.assert_allclose(later["post"][:,:,:16],expected["post"][:,:,:16],atol=1e-14)

    def test_no_alarm_changes_for_policy_and_exact_after_alarm_fallback(self):
        src = self.examples["A0"].take(slice(0,4))
        simple = play(self.p,src,("V1a",))
        risk = make_risk({"V1.3":1.},.1,16)
        trials = play(self.p,src,("V1a","V1.3"),risk)
        np.testing.assert_allclose(simple["log_bf"],trials["log_bf"],atol=2e-10)
        stop = np.maximum.accumulate(trials["log_bf"]>=-np.log(self.settings.event_false_alarm),axis=1)
        np.testing.assert_array_equal(trials["alpha"][:,1][stop],1.)


if __name__=="__main__":
    unittest.main(verbosity=2)
