"""Regression tests for mathematical identities and executable SSCD boundaries."""
import unittest
import numpy as np
from scipy.linalg import null_space
import sscd
from continuous_design import (_residual_covariance_polynomials,state_risk_polynomials,
    screen_continuum_2d,_poly_value)


class SignedDesignTests(unittest.TestCase):
    def test_signed_sampler_stationary_marginals_and_memory(self):
        for theta in (-2.,2.):
            beta=sscd.sample_signed_watermark_pairs(np.random.default_rng(310000+int(theta)),30000,2,3,theta)
            for time in range(2):
                self.assertLess(float(abs(beta[:,time].mean(axis=0)).max()),.02)
                np.testing.assert_allclose(beta[:,time].T@beta[:,time]/len(beta),.5*np.eye(3),atol=.025)
            directions=beta/np.linalg.norm(beta,axis=2,keepdims=True)
            empirical=float(np.mean(np.sum(directions[:,0]*directions[:,1],axis=1)**2))
            self.assertLess(abs(empirical-sscd.watson_stats(3,theta)['c']),.012)

    def test_compact_domain_and_iid(self):
        self.assertEqual(sscd.watson_stats(2,0)['mi_nats'],0.)
        self.assertEqual(sscd.dependence_interval(2,5,0.)['zeta_interval'],[0.,0.])
        with self.assertRaises(ValueError):sscd.theta_from_zeta(2,.99999)
        with self.assertRaises(ValueError):sscd.watson_stats(2,21.)

    def test_psd_checks(self):
        with self.assertRaises(np.linalg.LinAlgError):sscd.gaussian_covariance_kl_proxy(-np.eye(2),np.eye(2))
        with self.assertRaises(ValueError):sscd._worst_energy_risk(-np.eye(2),np.eye(2),10.)
        with self.assertRaises(ValueError):sscd._worst_energy_risk(np.eye(2),np.zeros((2,2)),10.)

    def test_general_measurement_dimension_finite_api(self):
        m=3;A=1.04*np.eye(m);T=[np.eye(m),A];G=.25*np.eye(m);F=.5*A
        K=[np.vstack([G,F@G]),np.vstack([np.zeros((m,m)),G])];W=np.eye(2*m)/2
        E=np.eye(m)+A.T@A;Z=np.hstack([-A,np.eye(m)]);theta=(2+1.04**2)*np.eye(m)
        Q=sscd.common_g_risk_matrix(T,K,W,0.,.5,.5)
        budget=sscd._worst_energy_risk(Q,E,10.)
        templates=[]
        for v in np.eye(m):
            x=v*np.sqrt(10/(v@E@v));templates.append((np.stack([t@x for t in T]),Z,theta))
        result=sscd.optimize_signed_zeta(m,templates,lambda z:sscd.common_g_risk_matrix(T,K,W,z,.5,.5),
                                         E,10.,.1,budget,.5,grid_size=17)
        self.assertTrue(result['best']['feasible'])
        self.assertIn('finite-template',result['status'])

    def test_m3_measurements_p2_continuum(self):
        A=1.04*np.eye(2);H=np.array([[1.,0.],[0.,1.],[1.,1.]])/np.sqrt(2)
        T=[H,H@A];Z=null_space(np.vstack(T).T).T
        process=np.vstack([np.zeros((3,2)),H]);V=process@process.T+np.eye(6)
        theta=Z@V@Z.T;E=sum(t.T@t for t in T)
        G=.2*np.linalg.pinv(H);F=.5*A
        K=[np.vstack([G,F@G]),np.vstack([np.zeros((2,3)),G])];W=np.eye(4)/2
        coeff=_residual_covariance_polynomials(T,Z,theta,E,10.,.5)
        Q=state_risk_polynomials(T,K,W)
        budget=sscd._worst_energy_risk(Q[0],E,10.)
        result=screen_continuum_2d(coeff,Q,E,10.,.1,budget,2,measurement_dimension=3,
                                   angle_points=128,zeta_tolerance=.005,evaluation_cap=100)
        self.assertTrue(result['best']['feasible'])
        self.assertLessEqual(result['best']['dependence'],.1+1e-10)

    def test_p3_not_silently_called_continuum(self):
        with self.assertRaises(ValueError):
            _residual_covariance_polynomials([np.eye(3)]*2,np.hstack([-np.eye(3),np.eye(3)]),np.eye(3),np.eye(3),10.,.5)


if __name__=='__main__':unittest.main(verbosity=2)
