"""Deterministic nominal frontier and an exact frozen-decimal infeasibility check."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfc
import sympy as sp

from frozen_inputs import (BASELINE_COMMIT, BUDGETS, DEFAULT_BASELINE, HERE, MODELS,
                           MODEL_PATH, TrustedConditioning, digest, load_prepared, setup, write_json)


def ball_projection(point, center, radius):
    if radius<0:
        raise ValueError("Radius must be nonnegative.")
    difference = point-center
    length = np.linalg.norm(difference,axis=-1,keepdims=True)
    scale = np.divide(radius,length,out=np.ones_like(length),where=length>0).clip(0.,1.)
    return center+scale*difference


def reachable_ball(preview, desired, center, radius, gain):
    """Project onto (preview + image(gain)) intersect ball; return the shift in packet space."""
    if radius<0:
        raise ValueError("Radius must be nonnegative.")
    U, sigma, Vt = np.linalg.svd(gain,full_matrices=False)
    active = sigma > np.finfo(float).eps*max(gain.shape)*sigma[0]
    span = U[:,active]
    def project(x):
        return (x @ span) @ span.T
    displacement = preview-center
    perpendicular = displacement-project(displacement)
    distance = np.linalg.norm(perpendicular,axis=-1)
    if np.any(distance>radius+2e-13):
        raise ValueError("The gain image cannot reach the given ball.")
    remaining = np.sqrt(np.maximum(0.,(radius-distance)*(radius+distance)))
    projected = project(desired-center)
    length = np.linalg.norm(projected,axis=-1)
    fraction = np.divide(remaining,length,out=np.ones_like(length),where=length>0).clip(0.,1.)
    result = center+perpendicular+fraction[...,None]*projected
    packet_shift = ((result-preview) @ span / sigma[active]) @ Vt[active]
    return result,packet_shift


def angular_excess(radius, eigenvalues):
    """E[(|N(0,C)|-r)_+^2] for the two-dimensional trusted/full mean gap."""
    left,right = np.maximum(eigenvalues,0.)
    if radius==0:
        return float(left+right)
    if left==right==0:
        return 0.
    def integrand(angle):
        variance = left*np.cos(angle)**2+right*np.sin(angle)**2
        if variance==0:
            return 0.
        ratio = radius/np.sqrt(2*variance)
        return max(0.,2*variance*np.exp(-ratio*ratio)-radius*np.sqrt(2*np.pi*variance)*erfc(ratio))
    return quad(integrand,0,np.pi/2,epsabs=2e-13,epsrel=5e-12)[0]*2/np.pi


def covariances(p):
    trust = TrustedConditioning(p)
    full = np.stack([prefix.conditional_state_cov for prefix in p.prefixes])
    cov_gap = trust.cov_x-full
    eig = np.linalg.eigvalsh(cov_gap)
    if np.min(eig)<-1e-12:
        raise ArithmeticError("The full packet set cannot increase conditional covariance.")
    return full,trust.cov_x,eig


def rms_cost(radius, full_trace, eigenvalues):
    return np.asarray(full_trace)+np.asarray([angular_excess(radius,pair) for pair in eigenvalues])


def radius_for_cost(maximum, full_trace, eigenvalues):
    mean_floor = float(np.mean(full_trace))
    trust = mean_floor+float(np.mean(eigenvalues.sum(axis=1)))
    if maximum<mean_floor:
        raise ValueError("The accuracy requirement is below the full-information floor.")
    if maximum==mean_floor and trust>mean_floor:
        return np.inf
    if maximum>=trust:
        return 0.
    ceiling = 1.
    while float(rms_cost(ceiling,full_trace,eigenvalues).mean())>maximum:
        ceiling *= 2
    return brentq(lambda r:rms_cost(r,full_trace,eigenvalues).mean()-maximum,
                  0.,ceiling,xtol=4e-13)


def base_rational(value):
    return sp.Matrix([[sp.Rational(cell) for cell in row] for row in value])


def _positive_definite_2(matrix):
    return matrix.shape==(2,2) and matrix==matrix.T and matrix[0,0]>0 and matrix.det()>0


def guaranteed_rayleigh_excess():
    # Uniform standard-deviation lower bound 4/25 and radius upper bound 4/25.
    scale = Fraction(4,25)
    steps = 96
    subdivisions = 16
    products = 128
    lower = Fraction(0)
    for j in range(1,steps+1):
        u = 1+Fraction(j,subdivisions)
        tick = u*u/2
        assert tick<products
        gain = scale*scale*Fraction(2*j-1,subdivisions**2)
        lower += gain*(1-tick/products)**products
    return lower


def rounded_exact(value, up=False, places=8):
    value = Fraction(int(value.numerator),int(value.denominator))
    scale = 10**places
    scaled = value*scale
    bound = -((-scaled.numerator)//scaled.denominator) if up else scaled.numerator//scaled.denominator
    return bound/scale


def exact_certificate(path, name, settings, p):
    """Check C >= (4/25)^2 I and cost > health cap using stored decimal data as rationals."""
    document = json.loads(path.read_text(encoding="utf-8"),parse_float=str)
    model = document["models"][name]
    A,Q = base_rational(model["A"]),base_rational(document["Q"])
    recv,target,neighbor = settings.receiver,settings.target,settings.healthy
    H,R = [list(map(base_rational,document[field])) for field in ("H","R")]
    node = model["nodes"][recv]
    K,G = (base_rational(node[field]) for field in ("K","G"))
    eye = sp.eye(2)
    J = eye-K@H[recv]-G@(H[neighbor]+H[target])
    Rg = K@R[recv]@K.T+G@(R[neighbor]+R[target])@G.T
    info_trusted = sum((H[j].T@R[j].inv()@H[j] for j in (recv,neighbor)),sp.zeros(2))
    info_full = info_trusted+H[target].T@R[target].inv()@H[target]
    Pt = Pf = P0 = eye
    stages = []
    exact_full,exact_V0 = sp.S(0),sp.S(0)
    sigma = sp.Rational(4,25)
    for k in range(settings.horizon):
        tr = (Pt.inv()+info_trusted).inv()
        full = (Pf.inv()+info_full).inv()
        v0 = J@P0@J.T+Rg
        assert _positive_definite_2(tr-full-sigma**2*eye),f"{name} time {k} covariance certificate"
        exact_full += sp.trace(full)
        exact_V0 += sp.trace(v0)
        stages.append((np.asarray(tr,dtype=float),np.asarray(full,dtype=float),float(sp.trace(v0))))
        Pt,Pf,P0 = [A@cov@A.T+Q for cov in (tr,full,v0)]
    t = settings.horizon
    full_mean,v0_mean = exact_full/t,exact_V0/t
    penalty = sp.Rational(guaranteed_rayleigh_excess())
    floor = full_mean+penalty
    upper = sp.Rational(51,50)*v0_mean
    if floor<=upper:
        raise ArithmeticError("The rational lower bound does not separate the 2% cap.")
    assert G.det()!=0
    f,tt,_ = covariances(p)
    np.testing.assert_allclose(np.stack([x[0] for x in stages]),tt,atol=4e-13,rtol=5e-13)
    np.testing.assert_allclose(np.stack([x[1] for x in stages]),f,atol=4e-13,rtol=5e-13)
    np.testing.assert_allclose([x[2] for x in stages],p.nominal_mse(),atol=4e-13,rtol=5e-13)
    return {"stages_with_verified_min_eigenvalue_bound":t,"sigma_lower_exact":"4/25",
            "r_upper_exact":"4/25","gain_det_exact_nonzero":True,
            "gain_det_exact":str(G.det()),
            "rayleigh_lower_bound_rounded_down":rounded_exact(penalty),
            "mean_health_lower_bound_rounded_down":rounded_exact(floor),
            "2pct_cap_rounded_up":rounded_exact(upper,up=True),
            "strict_margin_rounded_down":rounded_exact(floor-upper),
            "verdict":"infeasible for the stated shared trusted-center ball"}


def calculate(base):
    common,settings,spec,_ = setup(base)
    assert settings.horizon==32 and spec["danger_squared"]==.1
    proof_radius = float(np.sqrt(spec["danger_squared"])/2)
    assert sp.Rational(1,40)<sp.Rational(4,25)**2
    path = base/MODEL_PATH
    report = {"baseline":BASELINE_COMMIT,"model_sha256":digest(path),"steps":settings.horizon,
              "danger_squared":spec["danger_squared"],"diameter_radius_at_equality":proof_radius,
              "finite_sample_evaluation_started":False,"models":{}}
    for name in MODELS:
        p = load_prepared(base,name,common,settings)
        full,trust,eig = covariances(p)
        tr = np.trace(full,axis1=1,axis2=2)
        cost_at = lambda r:float(rms_cost(r,tr,eig).mean())
        v0 = float(p.nominal_mse().mean())
        radii = [radius_for_cost(v0*(1+eps),tr,eig) for eps in BUDGETS]
        rows = []
        for radius in sorted({*np.linspace(0,.6,61),proof_radius,*radii}):
            cost = cost_at(radius)
            rows.append({"radius":float(radius),"mse":cost,"pct_vs_V0":100*(cost/v0-1)})
        at = rms_cost(proof_radius,tr,eig)
        info = {"v0_mse":v0,"full_MMSE":float(tr.mean()),"trusted_MMSE":float(np.trace(
                    trust,axis1=1,axis2=2).mean()),"ball_min_at_boundary":float(at.mean()),
                "ball_min_at_boundary_pct_vs_V0":float(100*(at.mean()/v0-1)),
                "2pct_cap":1.02*v0,"minimum_radii":[{"health_target_pct":100*eps,"radius":r,
                     "diameter_squared":4*r*r} for eps,r in zip(BUDGETS,radii)],
                "state_G_singular_values":np.linalg.svd(p.model.G[settings.receiver],compute_uv=False).tolist(),
                "per_stage":{"full_trace":tr.tolist(),"C_eigenvalues":eig.tolist(),
                             "minimum_at_boundary":at.tolist()},
                "exact_certificate":exact_certificate(path,name,settings,p),"radius_profile":rows}
        assert rows[0]["mse"]>=at.mean()>info["2pct_cap"]
        report["models"][name] = info
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=DEFAULT_BASELINE)
    args = parser.parse_args()
    result = calculate(args.baseline)
    (HERE/"results").mkdir(exist_ok=True)
    write_json(HERE/"results/frontier.json",result)
    for name,row in result["models"].items():
        print(name,row["ball_min_at_boundary"],row["2pct_cap"],row["exact_certificate"]["verdict"])
