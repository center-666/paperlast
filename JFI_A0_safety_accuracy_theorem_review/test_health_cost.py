import unittest

import numpy as np

from archived_statistics import binomial_interval
from certified_cost import HealthCost, outward, real
from certified_first_packet import Band, cells, point_norm, point_projection
from health_cost_analysis import HERE, read, verify


class HealthCostChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = HERE.parent
        spec = read(cls.root/"JFI_reconstruction_attack_aware_soft_accommodation_V1/registration.json")
        path = cls.root/"common_g_reference_revision_20261115/inputs/five_node_aggregate_model.json"
        cls.costs = {n:HealthCost(path,n,spec) for n in ("unstable","stable")}

    def test_frozen_inputs(self):
        self.assertEqual(verify(self.root),54)

    def test_critical_cost_contains_expected_values(self):
        critical = real("1/10").sqrt()/2
        target = ("7.07054638072956828737","6.86513362513241041128")
        for cost, reference in zip(self.costs.values(),target,strict=True):
            result = 100*(cost.at(critical)/cost.v0-1)
            self.assertTrue(result.contains(real(reference)))
            self.assertTrue(result>2)

    def test_decreasing_and_convex_at_interior_points(self):
        for cost in self.costs.values():
            earlier,middle,later = [cost.at(r) for r in (.125,.25,.375)]
            self.assertTrue(earlier>middle>later>cost.full)
            self.assertTrue(2*middle<earlier+later)

    def test_probability_and_binomial_endpoints(self):
        low,high = binomial_interval(0,64,.05)
        self.assertEqual(low,0)
        self.assertLess(high,.06)
        self.assertGreater(high,.05)
        low,high = binomial_interval(31,64,.05)
        self.assertLess(low,31/64)
        self.assertGreater(high,31/64)

    def test_box_mass_and_midpoint_projection(self):
        coords, mass = cells(np.array([-1.,0.,1.]),np.array([-1.,0.,1.]),2)
        expected = real(".5").sqrt().erf()**2
        lo,hi = outward(expected)
        self.assertLessEqual(mass.lo.sum(),hi)
        self.assertGreaterEqual(mass.hi.sum(),lo)
        x = np.array([[0.,0.],[.3,0.],[-.4,0.]])
        np.testing.assert_allclose(point_projection(x,point_norm(x),.25),[[0,0],[.25,0],[-.25,0]])
        self.assertGreaterEqual(Band(-1.,2.).squared().hi,4)
        self.assertEqual(coords.lo.shape,(4,2))


if __name__=="__main__":
    unittest.main()
