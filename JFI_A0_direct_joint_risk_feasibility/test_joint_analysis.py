import unittest

import numpy as np

from exact_bounds import START, LAST
from joint_integral import LOG_THRESHOLD, Prefix, ball, prefix_events, work_posterior


class JointEventChecks(unittest.TestCase):
    def test_strict_alarm_timing(self):
        log_bf = np.zeros((4, LAST+1))
        log_bf[0, START-1] = LOG_THRESHOLD
        log_bf[1, START] = LOG_THRESHOLD
        log_bf[2, START+1] = LOG_THRESHOLD
        b = np.zeros((4, 1, 3, 2))
        b[:, 0, :, 0] = 0.4
        _, cumulative = prefix_events(log_bf,b)
        np.testing.assert_array_equal(cumulative[:, 0], [
            [False,False,False], [False,False,False],
            [True,True,True], [True,True,True],
        ])

    def test_late_danger_does_not_retroactively_count(self):
        bf = np.zeros((2, LAST+1))
        bf[0, START+1] = LOG_THRESHOLD
        b = np.zeros((2, 1, 3, 2))
        b[:, 0, 2, 0] = 0.4
        event, cumulative = prefix_events(bf,b)
        np.testing.assert_array_equal(cumulative, event)
        np.testing.assert_array_equal(cumulative[:, 0], [
            [False,False,False], [False,False,True],
        ])

    def test_single_prefix_bayes_integral(self):
        x = np.diag([0.5, 2.])
        t = np.array([0.7, -1.2])
        posterior = work_posterior(x[None,None,None], Prefix(np.eye(2),np.eye(2),np.eye(2)), t[None])
        model_covariance = np.eye(2)+x@x.T
        bf = np.exp(0.5*t@(np.eye(2)-np.linalg.inv(model_covariance))@t)/np.sqrt(
            np.linalg.det(model_covariance)
        )
        probability = bf/(63+bf)
        np.testing.assert_allclose(np.exp(posterior.log_bf), [(31+bf)/32])
        np.testing.assert_allclose(posterior.active_probability, [probability])
        np.testing.assert_allclose(posterior.correction, (probability*(np.eye(2)-np.linalg.inv(model_covariance))@t)[None])

    def test_projection_degenerate_cases_and_ellipse_bound(self):
        x = np.array([[0.,0.], [1.,2.], [1.,0.], [0.05,-0.08]])
        y = np.array([[0.,0.], [1.,2.], [-1.,0.], [0.08,1.]])
        previous = np.zeros(len(x))
        for r in (0.05, 0.1, np.sqrt(0.1)/2, 0.34, 1., 5.):
            distance = np.linalg.norm(ball(y,r)-ball(x,r),axis=-1)
            scale = 2*r/(np.maximum(r,np.linalg.norm(x,axis=-1))+np.maximum(r,np.linalg.norm(y,axis=-1)))
            self.assertTrue(np.all(previous <= distance+1e-14))
            self.assertTrue(np.all(distance <= scale*np.linalg.norm(x-y,axis=-1)+1e-14))
            previous = distance


if __name__ == "__main__":
    unittest.main()
