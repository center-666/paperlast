"""Checks for probability, geometry, and report invariants."""
import unittest

import numpy as np
from scipy.integrate import quad

from frontier import angular_excess, ball_projection, radius_for_cost, reachable_ball
from frozen_inputs import HERE, MODELS, load_json


class FrontierChecks(unittest.TestCase):
    def test_isotropic_penalty(self):
        for radius in (0.,.158,.25):
            sigma = .17
            integrated = quad(lambda x:(sigma*x-radius)**2*x*np.exp(-x*x/2),
                              radius/sigma,np.inf,epsabs=3e-13)[0]
            self.assertAlmostEqual(integrated,angular_excess(radius,[sigma*sigma]*2),places=12)

    def test_common_gain_image(self):
        center = np.array([.2,-.1])
        prior = np.array([0.,.5])
        desired = np.array([.9,-.3])
        gain = np.array([[.2,.4,.3],[-.1,-.5,.1]])
        result,shift = reachable_ball(prior,desired,center,.158,gain)
        np.testing.assert_allclose(prior+gain @ shift,result,atol=2e-14)
        np.testing.assert_allclose(result,ball_projection(desired,center,.158))
        rank_one = np.array([[1.,0.],[0.,0.]])
        with self.assertRaises(ValueError):
            reachable_ball(prior,desired,center,.5,rank_one)
        result,shift = reachable_ball(prior,desired,center,.625,rank_one)
        np.testing.assert_allclose(prior+rank_one @ shift,result)
        self.assertAlmostEqual(result[1],prior[1],places=12)
        self.assertAlmostEqual(np.linalg.norm(result-center),.625,places=12)

    def test_same_center_diameter_bound(self):
        center = np.array([.08,-.06])
        r = .158
        a = ball_projection(center+np.array([1.,0.]),center,r)
        b = ball_projection(center-np.array([1.,0.]),center,r)
        self.assertAlmostEqual(np.linalg.norm(a-b)**2,4*r*r)
        self.assertLess(float(np.linalg.norm(a-b)**2),.1)

    def test_standard_state_innovation_saturation_equivalent(self):
        center = np.array([[.3,.2],[.2,-.4],[0.,0.]])
        innovation = np.array([[.07,0.],[.3,-.4],[0.,0.]])
        radius = .158
        shrinkage = np.array([1.,radius/.5,1.])
        baseline = center+innovation*shrinkage[:,None]
        np.testing.assert_allclose(ball_projection(center+innovation,center,radius),
                                   baseline,atol=2e-15)

    def test_root_and_certificate_rows(self):
        data = load_json(HERE/"results/frontier.json")
        for model in MODELS:
            row = data["models"][model]
            proof = row["exact_certificate"]
            self.assertGreaterEqual(proof["2pct_cap_rounded_up"],row["2pct_cap"]-1e-15)
            self.assertGreater(proof["mean_health_lower_bound_rounded_down"],row["2pct_cap"])
            self.assertTrue(proof["gain_det_exact_nonzero"])
            p = row["per_stage"]
            root = radius_for_cost(row["2pct_cap"],p["full_trace"],np.array(p["C_eigenvalues"]))
            self.assertAlmostEqual(root,row["minimum_radii"][-1]["radius"],places=12)
            self.assertGreater(root,data["diameter_radius_at_equality"])

if __name__=="__main__":
    unittest.main()
