"""Verify directional moment restrictions and bounds on the frozen defensive model."""

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_solve
import sympy as sp

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "JFI_defensive_offline_validation"))
from replay_evaluator import MODEL, OBSERVATIONS, V1, load_receiver_modules, read_json, verify_inputs


def freeze(reg):
    parents = {**reg["frozen"]}
    p = BASE / "JFI_watermark_assisted_fusion_feasibility_20261119" / "replay_helpers_sha256.json"
    # Verify the parent manifest before trusting its entries.
    assert sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == parents[str(
        p.relative_to(BASE).as_posix())]
    parents.update(read_json(p)["files"])
    for name, expected in parents.items():
        assert sha256((BASE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, name
    return len(parents.keys() | verify_inputs(BASE)["files"].keys())


def symmetric_vector(M):
    """Coefficients of (cos 2theta, sin 2theta) for the trace pairing."""
    return np.array([M[0, 0]-M[1, 1], 2*M[0, 1]])


def exact_resource_determinants(record):
    """Use the original decimal model as an exact rational physical input."""
    rows = []
    for name, case in record["models"].items():
        A = sp.Matrix(case["A"])
        Id = sp.eye(A.rows)
        S = Id
        Q = sp.Matrix(record["Q"])
        C = sp.Matrix(record["H"][record["attacked_link"]["sender"]])
        R = sp.Matrix(record["R"][record["attacked_link"]["sender"]])
        assert S.rows == 2
        measurements = []
        for k in range(3):
            U = C*S*C.T+R
            measurements.append([U[0, 0]-U[1, 1], 2*U[0, 1]])
            S = A*S*A.T+Q
        exact = sp.Matrix(measurements[1:]).det()
        assert exact != 0, name
        rows.append({"model": name, "tick0_orientation_vector": [str(x) for x in measurements[0]],
                     "tick1_tick2_orientation_vectors": [[str(x) for x in v] for v in measurements[1:]],
                     "determinant_exact": str(exact), "determinant": float(exact)})
    return rows


def exact_fourth_map():
    """Algebra only. The orientation moments below are formal variables, not a trial encoder."""
    C0, S0, c4, s4, x, y, z = sp.symbols("c s c4 s4 x y z", real=True)
    M = sp.Matrix([[x,y],[y,z]])
    W = sp.Matrix([[C0,S0],[S0,-C0]])
    # Expectations of the quadratic double-angle terms depend on the fourth mode.
    averaged = (W*M*W).applyfunc(lambda f: sp.expand(f).subs({
        C0**2:(1+c4)/2, S0**2:(1-c4)/2,C0*S0:s4/2}, simultaneous=True))
    want = sp.trace(M)/2*sp.eye(2) + sp.Matrix(
        [[c4*(x-z)+s4*2*y, s4*(x-z)-c4*2*y],
         [s4*(x-z)-c4*2*y, -c4*(x-z)-s4*2*y]])/2
    assert (averaged-want).applyfunc(sp.simplify)==sp.zeros(2)

    q = Fraction(1,2)
    lam = Fraction(1,2)
    baseline_scale = (lam+q)**2 + q**2
    trace_scale = q**2
    # Apply the isotropic second moment map to the identity.
    assert baseline_scale + 2*trace_scale == Fraction(7,4)
    # In the isotropic known-onset reference the bias matrix is f0(I-P)+f1 P.
    # Check that its angle-mean depends on the second moment only.
    f0, f1 = sp.symbols("f0 f1", real=True)
    projector = (sp.eye(2) + W)/2
    bias = f0*(sp.eye(2)-projector) + f1*projector
    assert (bias.subs({C0:0,S0:0})-(f0+f1)*sp.eye(2)/2).applyfunc(sp.simplify)==sp.zeros(2)
    return {"matrix_identity": "Formal 2x2 fourth-moment operator verified",
            "isotropic_oracle_bias_average": "spectral-projector identity verified",
            "original_Gaussian_trace_map_coefficients": [str(baseline_scale),str(trace_scale)]}


def stats(Q, a, lam, q):
    a2 = np.dot(a,a)
    iso = (lam+q)**2+q*q
    base = iso*float(a@Q@a)+q*q*float(np.trace(Q))*a2
    spread = q*q*np.linalg.norm(symmetric_vector(Q))*a2
    assert base > 0 and spread >= 0
    return {"baseline_diagonal_second_moment": base, "absolute_variation_upper": float(spread),
            "percent_of_baseline": float(100*spread/base),
            "condition_number": float(np.linalg.cond(Q))}


def numerical(reg):
    _, common, evidence = load_receiver_modules(BASE)
    spec = read_json(BASE / V1 / "registration.json")
    s = evidence.Settings.from_registration(spec)
    assert spec["evaluation_onset"] == reg["first_packet"]
    lam, q = [spec["watermark"][k] for k in ("lambda","q")]
    assert lam == q == .5
    values = {}
    with np.load(BASE / OBSERVATIONS, allow_pickle=False) as archive:
        for name in reg["models"]:
            p = evidence.PreparedModel(common.load_model(name, BASE / MODEL), s)
            G = p.model.G[s.receiver]
            impact = G.T@G
            code = archive[f"{name}__original_Gaussian__ordinary__attack"]
            assert np.isclose(np.sum(code*code), s.coded_energy_reference, rtol=0, atol=3e-11)
            assert np.all(code[:reg["first_packet"]]==0)
            rows = []
            for age in reg["audited_packet_ages"]:
                k = reg["first_packet"] + age - 1
                size, width = (k+1)*p.m, p.m
                L = p.prefixes[k].target_cholesky
                precision = cho_solve((L,True),np.eye(size))
                np.testing.assert_allclose(precision, precision.T, atol=2e-12, rtol=2e-12)
                a = code[k]
                P0 = p.parameter_prior_cholesky @ p.parameter_prior_cholesky.T
                encoded_prior = p.model.H[s.target] @ P0 @ p.model.H[s.target].T
                current = stats(precision[-width:,-width:], a, lam, q)
                energy = sum(stats(precision[j*width:(j+1)*width,j*width:(j+1)*width],
                                   code[j],lam,q)["baseline_diagonal_second_moment"]
                             for j in range(reg["first_packet"],k+1))
                cov_diag = sum(q*q*(np.trace(precision[j*width:(j+1)*width,j*width:(j+1)*width])
                                   *np.dot(code[j],code[j])+code[j]@precision[j*width:(j+1)*width,
                                                                         j*width:(j+1)*width]@code[j])
                               for j in range(reg["first_packet"],k+1))
                mean = ((lam+q)*code[:k+1]).ravel()
                total_base = mean@precision@mean+cov_diag
                direct_map = sum(stats(precision[j*width:(j+1)*width,j*width:(j+1)*width],
                                       code[j],lam,q)["absolute_variation_upper"]
                                 for j in range(reg["first_packet"],k+1))
                if age == 1:
                    np.testing.assert_allclose(total_base,current["baseline_diagonal_second_moment"],
                                               rtol=3e-12,atol=2e-12)
                rows.append({
                    "packet_age": age, "absolute_tick": k,
                    "current_noise_precision_diagonal_block": precision[-width:,-width:].tolist(),
                    "raw_common_g_current": stats(impact,a,lam,q),
                    "current_information_part": current,
                    "expected_full_prefix_squared_distance_baseline": float(total_base),
                    "diagonal_only_baseline": float(energy),
                    "prefix_upper_change_with_independent_ticks": float(direct_map),
                    "prefix_upper_change_percent_of_baseline": float(100*direct_map/total_base),
                    "true_start_encoded_prior_covariance": encoded_prior.tolist(),
                    "prior_covariance_condition_number": float(np.linalg.cond(encoded_prior))})
            values[name]={"gain_gram":impact.tolist(),"registered_packet_results":rows}
    return values


def main():
    reg = read_json(HERE / "registration.json")
    count = freeze(reg)
    model = json.loads((BASE/MODEL).read_text(encoding="utf-8"),parse_float=sp.Rational)
    out = {"baseline":reg["baseline"],"frozen_input_count":count,
           "new_watermarks_or_paths":0,"resource_rank":exact_resource_determinants(model),
           "algebra":exact_fourth_map(),"numerical":numerical(reg)}
    assert freeze(reg)==count
    path = HERE / "direction_moment_results.json"
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    for n, data in out["numerical"].items():
        for x in data["registered_packet_results"]:
            print(n, x["packet_age"], f"prefix {x['prefix_upper_change_percent_of_baseline']:.5f}%",
                  f"instant {x['raw_common_g_current']['percent_of_baseline']:.5f}%")
    print(f"{count} unchanged frozen files; {len(out['resource_rank'])} exact ranks; no new watermarks or paths.")


if __name__ == "__main__":
    main()
