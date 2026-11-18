"""Independent receiver-only checks for the formulas and frozen records."""
import argparse
from pathlib import Path
import sys
import unittest

import numpy as np

from energy_inputs import (DEFAULT_BASELINE, HERE, MODELS, PREVIOUS, load_json, load_prepared,
                           setup, validate_spec)
from frozen_inputs import fresh_streams
from receiver import Method, collect
from inputs import Streams
from spectral_influence import (decoder_quadrature, decoded_square_moment, deterministic_open_loop_bound,
                                projected_moment, spectral_interval, spectral_quadrature)


BASE = DEFAULT_BASELINE


class EnergyChecks(unittest.TestCase):
    def test_gaussian_moments_identity_case(self):
        B = np.eye(2)[None]
        lam=q=.5
        # Norm squared for I*M is (lambda+R)^2, R exponential with mean one.
        integral = spectral_quadrature(B,lam,decoder_quadrature(1,q,2,14,809))
        self.assertLess(abs(integral-3.25),.014)
        np.testing.assert_allclose(projected_moment(lam,q,np.eye(2)),1.75*np.eye(2))
        self.assertEqual(decoded_square_moment(lam,q,2),1.75)
        self.assertAlmostEqual(deterministic_open_loop_bound(B,lam,q),1.75)
        lo,hi = spectral_interval(B,lam,q)
        self.assertLessEqual(lo,3.25)
        self.assertGreaterEqual(hi,3.25)

    def test_no_random_watermark_exact_consistency(self):
        B = np.array([[[1.2,.3],[-.4,.3]],[[.5,.4],[.1,.3]]])
        sq = np.linalg.norm(.5*B.transpose(1,0,2).reshape(2,4),ord=2)**2
        np.testing.assert_allclose(spectral_interval(B,.5,0.),[sq,sq],atol=2e-15)
        self.assertAlmostEqual(deterministic_open_loop_bound(B,.5,0.),sq)

    def test_exact_lower_outputs(self):
        observed = load_json(HERE/"results/affine_frontier.json")
        for name in MODELS:
            row = observed["models"][name]
            self.assertEqual(len(row["stages"]),32)
            self.assertGreater(row["no_danger_healthy_mse_lower_down"],row["nominal_two_pct_cap_up"])
            self.assertEqual(row["minimum_eligible_ticks_for_this_uniform_lower_certificate"],26)
            self.assertGreater(row["worst_total_danger_probability_lower_at_two_pct_down"],.009)
            self.assertLess(row["chance_feasible_upper_via_episode_mixture_up"],.90)

    def test_frozen_replay_bounds_and_coded_norm(self):
        data = load_json(HERE/"results/archived_comparison.json")
        self.assertEqual(data["live_adaptive_trials"],0)
        self.assertEqual(data["new_physical_seeds"],0)
        for name in MODELS:
            rows = data["models"][name]
            self.assertEqual(rows["profiles"]["healthy"]["maximum_episode_coded_energy"],0.)
            for case,row in rows["profiles"].items():
                self.assertLess(row["maximum_episode_coded_energy"],10.+2e-10)
                self.assertAlmostEqual(sum(row["mean_coded_energy_by_time"]),row["mean_episode_coded_energy"])
                self.assertLess(row["reconstruction_max_abs"],2e-12)
            for row in rows["selected_pair_upper_envelope_examples"].values():
                self.assertLessEqual(row["archived_episode_0_max_displacement"],
                                     row["archived_episode_0_max_bound"]+2e-12)

    def test_frozen_receiver_does_not_read_later_observations(self):
        common,settings,spec,_ = setup(BASE)
        validate_spec(settings,spec)
        info = load_json(PREVIOUS/"registration.json")["holdout"]
        p = load_prepared(BASE,"unstable",common,settings)
        # Splice the later part from another healthy record, for a prefix API unit test.
        archived = fresh_streams(BASE,p,spec,info["seed"]["unstable"],2,healthy_only=True)["healthy"]
        a,b = (archived.take(slice(i,i+1)) for i in (0,1))
        k = 10
        def join(field):
            return np.concatenate((getattr(a,field)[:,:k],getattr(b,field)[:,k:]),axis=1)
        alternate = Streams(a.state,join("trusted"),join("target"),join("decoder"),a.offset,a.start,a.stop)
        choices = (Method("V0","raw"),Method("V1a","structure"),
                   Method("trial-anchor","anchor",.25),Method("trial-dual","dual",.02))
        x,y = (collect(p,record,choices) for record in (a,alternate))
        np.testing.assert_allclose(x.post[:,:,:k],y.post[:,:,:k],atol=3e-13)
        np.testing.assert_allclose(x.log_bf[:,:k],y.log_bf[:,:k],atol=3e-13)


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args,unknown = parser.parse_known_args()
    BASE = args.baseline
    unittest.main(argv=[sys.argv[0],*unknown])
