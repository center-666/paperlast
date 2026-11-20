"""Reviewer checks of T1--T3. No physical trajectories or efficacy experiment.

The analytic proof is in mathematical_review.md. Deterministic numerical grids
check implementation and constants; they are not a proof over all realizations.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / "B1_three_hypothesis_feasibility_20261120"))
from experiment import read, model_set


def quantile(p):
    return .5 * chi2.ppf(p, 1)


def design(a, b, c, sign=1., lam=.5, ratio=14/13):
    p, q0, q1 = lam + a*a, a*sign*b, -a*sign*c
    M0 = np.array([[p, q0], [q0, lam+b*b]])
    M1 = np.array([[p, q1], [q1, lam+c*c]])
    B = np.array([[0., p, q0], [1., q0, lam+b*b],
                  [0., p, ratio*q1], [1., q1/ratio, lam+c*c]])
    return M0, M1, B


def inverse_rows(a, b, c, sign=1., lam=.5, ratio=14/13, threshold=.15):
    p, q0, q1 = lam + a*a, a*sign*b, -a*sign*c
    e = np.eye(4)
    w0, w1 = e[1] - q0/p*e[0], e[3] - q1/(ratio*p)*e[2]
    h0, h1 = lam+lam*b*b/p, lam+lam*c*c/p
    if abs(a)*(b+ratio*c) >= threshold:
        attack2 = (e[0]-e[2])/(a*sign*(b+ratio*c))
        branch = "difference_u"
    else:
        attack2 = (w1-w0)/(lam/p*(c*c-b*b))
        branch = "schur_difference"
    if b <= c:
        attack1, state = e[0]/p-q0/p*attack2, w0-h0*attack2
    else:
        attack1, state = e[2]/p-ratio*q1/p*attack2, w1-h1*attack2
    return np.vstack([state, attack1, attack2]), branch


def constants():
    lam, ratio, threshold = .5, 14/13, .15
    delta, upper = quantile(.45), quantile(.55)
    pmax = lam+threshold**2/delta
    c2 = max(np.sqrt(2)/threshold, pmax/(lam*delta)*np.sqrt(2+2*(threshold/lam)**2))
    c1 = 1/lam+ratio*np.sqrt(upper)/(2*np.sqrt(lam))*c2
    cx = np.sqrt(1+upper/(4*lam))+(lam+upper)*c2
    total = np.sqrt(c2*c2+c1*c1+cx*cx)
    return dict(lambda_=lam, ratio=ratio, threshold=threshold, delta=delta, B_squared=upper,
                pmax=pmax, C2=c2, C1=c1, Cx=cx, C=total, sigma_min_lower=1/total,
                rowwise_Frobenius_bound_squared=total*total,
                scalar_half_shift_variance_bound=1+2*(.5+quantile(.5))/quantile(.5)
                    +2*((.5+quantile(.5))/quantile(.5))**2,
                scalar_jitter_variance_bound=1+2*(.5+upper)/delta+2*((.5+upper)/delta)**2)


def deterministic_grid(bounds):
    records = []
    max_identity, max_frobenius, minimum_sigma = 0., 0., float("inf")
    ratio = bounds["ratio"]
    a_values = [0., 1e-12, -1e-9, .01, -.05, .15, -.3, 1., -3., 10., 100., 1e4]
    u_values = [1e-10, 1e-5, .01, .1, .225, .449999, .450001, .499, .55, .77, .99, 1-1e-10]
    branches = {}
    for a, u, shift, sign in product(a_values, u_values, [.45, .475, .5, .525, .55], [-1., 1.]):
        paired = (u+shift) % 1
        # Shift endpoints and a few u values can coincide exactly at a quantile zero;
        # zero is allowed, whereas quantile one is outside the finite support.
        b, c = np.sqrt(quantile(u)), np.sqrt(quantile(paired))
        M0, M1, B = design(a, b, c, sign)
        L, branch = inverse_rows(a, b, c, sign)
        err = float(np.linalg.norm(L@B-np.eye(3), ord=np.inf))
        inverse_frobenius = float(np.linalg.norm(L))
        sigma = float(np.linalg.svd(B, compute_uv=False)[-1])
        assert abs(c*c-b*b) >= bounds["delta"]-1e-12
        assert min(b*b, c*c) <= bounds["B_squared"]+1e-12
        assert err < 5e-7, (a, u, shift, sign, err)
        assert inverse_frobenius <= bounds["C"]+1e-9
        assert sigma >= bounds["sigma_min_lower"]-1e-7
        max_identity = max(max_identity, err)
        max_frobenius = max(max_frobenius, inverse_frobenius)
        minimum_sigma = min(minimum_sigma, sigma)
        branches[branch] = branches.get(branch, 0)+1
    # A large first-coordinate amplitude demonstrates why condition number is
    # not uniformly bounded, despite the proved inverse-norm bound.
    for a in (1., 10., 100., 1000.):
        _, _, B = design(a, np.sqrt(quantile(.2)), np.sqrt(quantile(.7)))
        sv = np.linalg.svd(B, compute_uv=False)
        records.append(dict(a=a, largest=float(sv[0]), smallest=float(sv[-1]), condition=float(sv[0]/sv[-1])))
    return dict(points=sum(branches.values()), branches=branches, left_inverse_max_error=max_identity,
                max_observed_frobenius=max_frobenius, minimum_observed_singular=minimum_sigma,
                condition_number_diagnostic=records, interpretation="Finite deterministic grid only; analytic bounds proved separately")


def exact_boundary_and_compensation_checks():
    A = F(28,25)
    enumerated = 0
    max_odd, max_even = 0, 0
    for onset in range(0, 13):
        for duration in range(1, 11):
            attack = [F(0)]*26
            for t in range(onset, onset+duration):
                attack[t] = A**(t-onset)*F(2,7)
            inconsistent = [t for t in range(1,26) if attack[t]-A*attack[t-1]]
            bad_blocks = [t//2 for t in range(1,26,2) if attack[t]-A*attack[t-1]]
            assert len(inconsistent) <= 2 and len(bad_blocks) <= 2
            if duration % 2 == 0:
                assert len(bad_blocks) == (2 if onset%2 else 0)
                max_even = max(max_even, len(bad_blocks))
            else:
                assert len(bad_blocks) == 1
                max_odd = max(max_odd, len(bad_blocks))
            enumerated += 1
    # An arbitrary 2x2 rational recursion confirms paired accumulation algebra.
    Fm = [[F(2,5),F(1,7)],[F(-1,9),F(1,3)]]
    G = [[F(1,2),F(1,5)],[F(0),F(2,3)]]
    mv = lambda M,v: [sum(M[i][j]*v[j] for j in range(2)) for i in range(2)]
    plus = lambda *vectors: [sum(v[j] for v in vectors) for j in range(2)]
    rprev=[F(3,4),F(-2,5)]
    eta0=[F(1,3),F(-1,2)]
    eta1=[F(2,9),F(4,7)]
    one = plus(mv(Fm,rprev),mv(G,eta0))
    two = plus(mv(Fm,one),mv(G,eta1))
    block = plus(mv(Fm,mv(Fm,rprev)),mv(Fm,mv(G,eta0)),mv(G,eta1))
    assert two == block
    # Projection has a healthy-risk guarantee but can destroy exact correction.
    raw, healthy_mean, candidate, budget = F(10),F(10),F(0),F(1)
    projected = F(9)
    assert (projected-healthy_mean)**2 == (raw-healthy_mean)**2+budget
    return dict(onset_duration_pairs=enumerated, max_bad_blocks_even_duration=max_even,
                max_bad_blocks_odd_duration=max_odd,
                block_recursion_exact=[str(x) for x in two],
                projection_counterexample=dict(true_state=0, raw=10, nominal_healthy_mean=10,
                    perfect_candidate=0, budget=1, projected=9, candidate_squared_error=0,
                    projected_squared_error=81, raw_squared_error=100))


def regional_noise_and_gain():
    reg=read(BASE/"B1_three_hypothesis_feasibility_20261120/registration.json")
    model=model_set(reg)[reg["additional_model"]]
    a1,a2=model.A[0,0],model.A[1,1]
    # u_t subtracts the same-time local first-coordinate observation. This
    # cancels x1 and its process noise instead of subtracting a noiseless oracle.
    covariance=np.diag([model.R[0][0,0]+model.R[1][0,0],model.R[0][1,1],
        (model.R[0][0,0]+model.R[1][0,0])/a1**2,(model.Q[1,1]+model.R[0][1,1])/a2**2])
    return dict(K=model.K[1].tolist(),G=model.G[1].tolist(),det_G=float(np.linalg.det(model.G[1])),
                explicit_normalized_noise_covariance=covariance.tolist(),
                noise_spectral_bound=float(np.linalg.eigvalsh(covariance)[-1]))


def main():
    bounds=constants()
    output=dict(status="passed", scope="Mathematical review and deterministic formula checks; no new physical experiment",
                constants=bounds, deterministic_grid=deterministic_grid(bounds),
                exact_boundary_and_compensation=exact_boundary_and_compensation_checks(),
                regional_noise_and_gain=regional_noise_and_gain())
    result=HERE/"checks/theory_checks.json"
    result.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(output,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
