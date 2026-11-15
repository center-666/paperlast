"""Independent G0 algebra and fixed-gain covariance checks; no sample generation."""
from pathlib import Path
import hashlib
import json

import numpy as np
import sympy as sp
from scipy.linalg import solve_discrete_lyapunov

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "original_estimator_gate_20261114" / "five_node_aggregate_model.json"


def exact_checks():
    a, q, r, c = sp.symbols("a q r c", real=True)
    # Order [w_0, v_1,0, v_2,0, v_1,1, v_2,1].  The two same-time
    # 2-sensor covariance blocks have fixed diagonal r and variable c.
    joint = sp.diag(q, r, r, r, r)
    joint[1, 2] = joint[2, 1] = c
    joint[3, 4] = joint[4, 3] = c
    time_parity = sp.Matrix([[1, -a, 0, 1, 0]])
    spatial = sp.Matrix([[0, 0, 0, 1, -1]])
    assert sp.simplify((time_parity * joint * time_parity.T)[0] - q - (a*a+1)*r) == 0
    assert sp.simplify((spatial * joint * spatial.T)[0] - 2*r + 2*c) == 0

    # Local plus sum of 2 neighbor packets cannot identify their individual
    # local-neighbor cross-correlations, even with all marginal variances fixed.
    eps = sp.Rational(1, 4)
    S = sp.Matrix([[1, 0, 0], [0, 1, 1]])
    R0 = sp.eye(3)
    R1 = sp.Matrix([[1, eps, -eps], [eps, 1, 0], [-eps, 0, 1]])
    assert R1.is_positive_definite
    assert S*R0*S.T == S*R1*S.T

    # Actual 5-node model: H0=I and H2=D.  Equal 2-packet autonomous
    # trajectories require the nonzero commutator to annihilate their seed.
    D = sp.diag(sp.Rational(6, 5), sp.Rational(7, 5))
    exact_models = {
        "unstable": sp.Matrix([[sp.Rational(108,100),sp.Rational(5,100)],
                               [sp.Rational(1,100),sp.Rational(105,100)]]),
        "stable": sp.Matrix([[sp.Rational(8,10),sp.Rational(5,100)],
                             [sp.Rational(1,100),sp.Rational(9,10)]]),
    }
    commutators = {}
    for name, A in exact_models.items():
        C = A*D-D*A
        assert C.det() == sp.Rational(1, 50000)
        T0 = sp.eye(2).col_join(A)
        T2 = D.col_join(D*A)
        assert T0.row_join(-T2).rank() == 4
        commutators[name] = {"matrix": str(C), "determinant": str(C.det()),
                              "joint_2_packet_rank": 4}
    return {
        "single_source_temporal_parity_variance": str(q+(a*a+1)*r),
        "two_source_spatial_difference_variance": str(2*r-2*c),
        "same_aggregate_distinct_raw_covariances": {
            "R0": str(R0), "R1": str(R1), "S": str(S),
            "shared_aggregate_covariance": str(S*R0*S.T),
            "R1_leading_principal_minors": [str(R1[:i,:i].det()) for i in range(1,4)],
        },
        "actual_model_two_packet_attribution_condition": commutators,
    }


def numerical_checks(model):
    Q = np.asarray(model["Q"], float)
    H = [np.asarray(x, float) for x in model["H"]]
    target = 1
    out = {}
    for name, entry in model["models"].items():
        A = np.asarray(entry["A"], float)
        nd = entry["nodes"][target]
        K, G = np.asarray(nd["K"], float), np.asarray(nd["G"], float)
        Hsum = sum((H[j] for j in nd["neighbors"]), start=np.zeros((2,2)))
        L = np.hstack([K,G])
        C = np.vstack([H[target],Hsum])
        J = np.eye(2)-L@C
        F = J@A
        assert np.max(np.abs(F-np.asarray(nd["posterior_F"]))) < 1e-15
        S = np.zeros((4,10))
        S[:2,2*target:2*target+2] = np.eye(2)
        for j in nd["neighbors"]:
            S[2:,2*j:2*j+2] = np.eye(2)
        rows=[]
        for correlation in [-.2, 0., .2]:
            fullR=.5*np.kron((1-correlation)*np.eye(5)+correlation*np.ones((5,5)),np.eye(2))
            assert np.linalg.eigvalsh(fullR)[0] > .09
            Rbar=S@fullR@S.T
            expected=np.block([[.5*np.eye(2),correlation*np.eye(2)],
                               [correlation*np.eye(2),(1+correlation)*np.eye(2)]])
            assert np.max(np.abs(Rbar-expected))<1e-15
            Omega=J@Q@J.T+L@Rbar@L.T
            P=solve_discrete_lyapunov(F,Omega)
            prior=A@P@A.T+Q
            assert np.max(np.abs(P-F@P@F.T-Omega))<2e-15
            if correlation==0:
                assert np.max(np.abs(P-np.asarray(nd["posterior_covariance"])))<2e-15
            rows.append({"equicorrelation":correlation,
                         "full_R_min_eigenvalue":float(np.linalg.eigvalsh(fullR)[0]),
                         "aggregate_R":Rbar.tolist(),
                         "posterior_trace":float(np.trace(P)),
                         "predictor_trace":float(np.trace(prior)),
                         "lyapunov_residual_max":float(np.max(np.abs(P-F@P@F.T-Omega)))})
        out[name]={"fixed_F":F.tolist(),
                   "fixed_F_spectral_radius":float(max(abs(np.linalg.eigvals(F)))),
                   "fixed_gains_not_recomputed":True,"correlation_cases":rows}
    return out


if __name__=="__main__":
    source_bytes=SOURCE.read_bytes()
    model=json.loads(source_bytes.decode("utf-8"))
    result={"scope":"Independent symbolic algebra and double precision fixed-gain Lyapunov diagnostics; no physical samples; no new gain design",
            "input":str(SOURCE),"input_sha256":hashlib.sha256(source_bytes).hexdigest(),
            "exact_symbolic_checks":exact_checks(),
            "floating_fixed_gain_checks":numerical_checks(model)}
    output=HERE/"independent_g0_results.json"
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(output),"sha256_input":result["input_sha256"],
                      "models":{k:{"rho":v["fixed_F_spectral_radius"],
                                   "cases":[{"r":r["equicorrelation"],"post":r["posterior_trace"],
                                             "prior":r["predictor_trace"]}
                                            for r in v["correlation_cases"]]}
                                for k,v in result["floating_fixed_gain_checks"].items()}},
                     ensure_ascii=False,indent=2))
