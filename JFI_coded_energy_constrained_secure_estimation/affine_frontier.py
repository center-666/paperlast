"""Exact finite-coded-energy lower certificates for healthy precision and total danger."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import sympy as sp

from energy_inputs import (BASE_COMMIT, DEFAULT_BASELINE, HERE, MODELS, MODEL_PATH, base_rational,
                           covariances, digest, load_prepared, rounded_exact, setup,
                           validate_spec, write_json)
from spectral_influence import coefficients


S_MIN = Fraction(51,250)
V_MIN = Fraction(3,5)
DECODER_FLOOR = Fraction(1,2)
GAMMA = Fraction(1,10)
RELAXED_GAMMA = Fraction(21,200)
ANGLE_FRACTION_LOWER = Fraction(19,100)


def positive_part(r):
    return max(Fraction(0),r)


def rounded_bounds(c,partitions=32,last=8,product=256):
    """Exact lower for E penalty, and its pointwise supremum."""
    other = positive_part(S_MIN-c/DECODER_FLOOR)**2
    def penalty(r):
        return V_MIN*(positive_part(S_MIN-c/(DECODER_FLOOR+r))**2+other)
    total = previous = penalty(Fraction(0))
    for j in range(1,partitions*last+1):
        x = Fraction(j,partitions)
        assert x<product
        current = penalty(x)
        total += (current-previous)*(1-x/product)**product
        previous = current
    highest = V_MIN*(S_MIN*S_MIN+other)
    return sp.Rational(total),sp.Rational(highest)


def psd_strict(A):
    return A.shape==(2,2) and A==A.T and A[0,0]>0 and A.det()>0


def raw_covariance_certificate(path,name,p):
    frozen = json.loads(path.read_text(encoding="utf-8"),parse_float=str)
    data = frozen["models"][name]
    settings = p.settings
    i,h,j = (settings.receiver,settings.healthy,settings.target)
    H,R = [list(map(base_rational,frozen[key])) for key in ("H","R")]
    A,Q = base_rational(data["A"]),base_rational(frozen["Q"])
    K,G = (base_rational(data["nodes"][i][key]) for key in ("K","G"))
    eye = sp.eye(2)
    J = eye-K@H[i]-G@(H[j]+H[h])
    noise = K@R[i]@K.T+G@(R[h]+R[j])@G.T
    prior = prior_0 = prior_T = eye
    trust_info = sum((H[l].T@R[l].inv()@H[l] for l in (i,h)),sp.zeros(2))
    target_info = H[j].T@R[j].inv()@H[j]
    values = []
    full_total = trust_total = base_total = sp.S(0)
    s2 = sp.Rational(S_MIN)**2
    v = sp.Rational(V_MIN)
    for k in range(p.settings.horizon):
        C = (prior.inv()+trust_info).inv()
        V = H[j]@C@H[j].T+R[j]
        B = C@H[j].T@V.inv()
        all_info = (C.inv()+target_info).inv()
        trusted_only = (prior_T.inv()+trust_info).inv()
        base = J@prior_0@J.T+noise
        assert psd_strict(B.T@B-s2*eye),f"{name} k={k}: current gain lower"
        assert psd_strict(V-v*eye),f"{name} k={k}: current innovation lower"
        assert B==all_info@H[j].T@R[j].inv()
        reference = coefficients(p,k)["full_nominal_conditional"][-1]
        np.testing.assert_allclose(np.asarray(B,dtype=float),reference,atol=2e-13,rtol=3e-13)
        Lkk = p.prefixes[k].target_cholesky[-p.m:,-p.m:]
        np.testing.assert_allclose(np.asarray(V,dtype=float),Lkk@Lkk.T,atol=2e-13,rtol=3e-13)
        np.testing.assert_allclose(float(sp.trace(base)),p.nominal_mse()[k],atol=2e-13,rtol=3e-13)
        assert G.det()!=0
        full_total += sp.trace(all_info)
        base_total += sp.trace(base)
        trust_total += sp.trace(trusted_only)
        prior,prior_0,prior_T = [A@x@A.T+Q for x in (all_info,base,trusted_only)]
        values.append({"time":k,"sigma_min_B":float(np.linalg.svd(reference,compute_uv=False)[-1]),
                       "lambda_min_V":float(np.linalg.eigvalsh(Lkk@Lkk.T)[0])})
    T = settings.horizon
    return full_total/T,base_total/T,trust_total/T,values


def calculate(base):
    common,settings,spec,_ = setup(base)
    validate_spec(settings,spec)
    path = base/MODEL_PATH
    penalty0,_ = rounded_bounds(GAMMA)
    penalty1,largest = rounded_bounds(RELAXED_GAMMA)
    # cos(19*pi/200) > 20/21, so a planar uniform angle exceeds the threshold
    # on strictly more than 19% of the angles when the norm is > 21/20 of the cap.
    upper_angle = Fraction(22,7)*ANGLE_FRACTION_LOWER/2
    assert 1-upper_angle**2/2>Fraction(20,21)
    results = {}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        full,raw,trust,steps = raw_covariance_certificate(path,name,p)
        cap = sp.Rational(51,50)*raw
        headroom = cap-full
        least = full+penalty0
        risk = sp.Rational(ANGLE_FRACTION_LOWER)*(penalty1-headroom)/largest
        switch = 1-headroom/(trust-full)
        assert least>cap and risk>0 and 0<switch<1
        min_eligible = min(n for n in range(1,33) if full+penalty0*n/32>cap)
        full_cond = covariances(p)[0]
        np.testing.assert_allclose(float(full),np.trace(full_cond,axis1=1,axis2=2).mean(),atol=2e-13)
        results[name] = {"stages":steps,"nominal_V0":float(raw),"nominal_full_MMSE":float(full),
                         "nominal_trusted_only":float(trust),
                         "nominal_two_pct_cap_up":rounded_exact(cap,up=True),
                         "no_danger_healthy_mse_lower_down":rounded_exact(least),
                         "no_danger_percent_lower_down":rounded_exact(100*(least/raw-1)),
                         "no_danger_margin_above_two_pct_down":rounded_exact(least-cap),
                         "minimum_eligible_ticks_for_this_uniform_lower_certificate":min_eligible,
                         "worst_total_danger_probability_lower_at_two_pct_down":rounded_exact(risk),
                         "chance_feasible_upper_via_episode_mixture_up":rounded_exact(switch,up=True)}
        print(name,results[name]["no_danger_percent_lower_down"],results[name][
              "worst_total_danger_probability_lower_at_two_pct_down"],flush=True)
    return {"baseline":BASE_COMMIT,"model_sha256":digest(path),"waveform_energy":10,
            "D_squared":.1,"comparison_sets":{"zero_danger":"every fixed admissible single-packet change at every allowed epoch",
            "nominal":"affine in current target given trusted history, past target, and synchronized watermark"},
            "exact_floors":{"sigma_min_B":str(S_MIN),"lambda_min_V":str(V_MIN),
                            "gamma":str(GAMMA),"relaxed_gamma":str(RELAXED_GAMMA),
                            "angle_fraction_lower":str(ANGLE_FRACTION_LOWER)},
            "penalty_lower_gamma_down":rounded_exact(penalty0),
            "penalty_lower_relaxed_gamma_down":rounded_exact(penalty1),
            "pointwise_relaxed_penalty_upper_up":rounded_exact(largest,up=True),
            "models":results}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    result = calculate(args.baseline)
    (HERE/"results").mkdir(exist_ok=True)
    write_json(HERE/"results/affine_frontier.json",result)
