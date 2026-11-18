"""Independent covariance, policy, and accounting checks on the frozen receiver."""
import argparse
from pathlib import Path
import sys
import unittest

import numpy as np

from v14_inputs import DEFAULT_BASELINE, MODELS, legacy_cases, load_prepared, setup
from dual_receiver import METHODS, OnlineDualReceiver, ledger_cap
from paired_replay import collect, evaluate
from trusted_prediction import TrustedConditioning
from accommodation import OnlineEvidence


class TestV14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common, settings, cls.spec, _ = setup(DEFAULT_BASELINE)
        cls.prepared = {m:load_prepared(DEFAULT_BASELINE,m,cls.common,settings) for m in MODELS}
        cls.cases = {m:legacy_cases(DEFAULT_BASELINE,p,cls.spec,cls.common)[1].copy()
                     for m,p in cls.prepared.items()}

    def test_conditional_trusted_moments_reproduced_with_kalman_recursion(self):
        for m,p in self.prepared.items():
            s,model = p.settings,p.model
            t = TrustedConditioning(p)
            H = np.vstack([model.H[s.receiver],model.H[s.healthy]])
            R = np.zeros((2*p.m,2*p.m))
            R[:p.m,:p.m] = model.R[s.receiver]
            R[p.m:,p.m:] = model.R[s.healthy]
            H_j = model.H[s.target]
            observations = self.cases[m]["A1_C_mismatch"].trusted[:3]
            x = np.zeros((3,p.n))
            P = np.eye(p.n)
            I = np.eye(p.n)
            for k in range(s.horizon):
                U = np.linalg.solve(H @ P @ H.T+R,H @ P).T
                x = x+(observations[:,k]-x @ H.T) @ U.T
                P = (I-U @ H) @ P @ (I-U @ H).T+U @ R @ U.T
                estimates = t.at(observations,k)
                np.testing.assert_allclose(
                    estimates.cov_y,H_j @ P @ H_j.T+model.R[s.target],atol=2e-12)
                np.testing.assert_allclose(estimates.cov_x,P,atol=2e-12)
                np.testing.assert_allclose(estimates.mean_x,x,atol=2e-12)
                np.testing.assert_allclose(estimates.mean_y,x @ H_j.T,atol=2e-12)
                x,P = x @ model.A.T,model.A @ P @ model.A.T+model.Q

    def test_ledger_root(self):
        c = np.array([0., 2., 2., 2., 2., 2., 2.])
        h = np.array([0., 0., 3., -3., -1., .001, -.001])
        a = np.array([0., 1., .5, 1., 0., .17, .17])
        cap = ledger_cap(c,h,a)
        np.testing.assert_allclose(cap[:3],[1.,np.sqrt(.5),1.])
        q = cap*(c*cap-2*h)
        assert np.all(q<=a+1e-13)
        np.testing.assert_allclose(q[cap<1],a[cap<1],atol=1e-13)

    def test_live_receiver_and_gls_against_frozen_paths(self):
        for name,p in self.prepared.items():
            for case in ("healthy","A0_onset_replay","A1_C_mismatch"):
                seq = self.cases[name][case].take(slice(0,1))
                result = collect(p,seq)
                live = OnlineDualReceiver(p,self.common.common_step)
                observer = OnlineEvidence(p,self.common.common_step)
                j = METHODS.index("V1.4")
                for k in range(p.settings.horizon):
                    frame = (seq.trusted[0,k,:p.m],seq.trusted[0,k,p.m:],
                             seq.target[0,k],seq.decoder[0,k])
                    a = live.step(*frame)
                    np.testing.assert_allclose(a.posterior,result["post"][0,j,k],atol=3e-10,rtol=3e-10)
                    np.testing.assert_allclose(a.w_S_T,result["weights"][0,j,k],atol=4e-10)
                    np.testing.assert_allclose(a.log_event_bf,result["log_bf"][0,k],atol=4e-10)
                    np.testing.assert_allclose(a.target_deviation,frame[2]-a.trusted_target,atol=1e-14)
                    b = observer.observe(*frame)
                    expected = (p.model.G[p.settings.receiver] @ b.fits[b.matched_start].gls_delta
                                if b.matched_alarm else np.zeros(p.n))
                    np.testing.assert_allclose(expected,result["change"][0,METHODS.index("GLS"),k],atol=2e-10)

    def test_two_world_and_healthy_ledger_telescope(self):
        for name,p in self.prepared.items():
            stream = self.cases[name]["A1_H_mismatch"].take(slice(0,3))
            reference = self.cases[name]["healthy"].take(slice(0,3))
            nominal, actual = collect(p,reference),collect(p,stream)
            evaluate(p,stream,nominal,actual,self.spec["danger_squared"])
            # Pure trusted estimators use the same observations in the two worlds.
            for method in ("trusted-MMSE","trusted-replacement"):
                j = METHODS.index(method)
                np.testing.assert_allclose(nominal["post"][:,j],actual["post"][:,j],atol=1e-12)
            allocation = .02*p.nominal_mse()
            T = p.settings.horizon
            law = p.model.F(p.settings.receiver)
            for key in ("V1.2a","trusted-budget","V1.4"):
                j = METHODS.index(key)
                np.testing.assert_allclose(nominal["remainder"][:,j],np.cumsum(
                    allocation-nominal["charge"][:,j],axis=1),atol=8e-12)
                realised = np.zeros(3)
                for k in range(T):
                    # Extend the existing prefix with no further correction.
                    u = nominal["post"][:,j,k]+nominal["change"][:,j,k]
                    delta = nominal["change"][:,j,k].copy()
                    for h in range(k,T):
                        if h>k:
                            v0 = nominal["post"][:,0,h]-nominal["post"][:,0,h-1] @ law.T
                            u = u @ law.T+v0
                            delta = delta @ law.T
                        err = u-reference.state[:,h]
                        realised += np.sum(delta*(delta-2*err),axis=1)
                squared = np.sum((nominal["post"]-reference.state[:,None])**2,axis=(2,3))
                np.testing.assert_allclose(realised,squared[:,j]-squared[:,0],atol=2e-11)


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args, unknown = parser.parse_known_args()
    DEFAULT_BASELINE = args.baseline
    unittest.main(argv=[sys.argv[0],*unknown])
